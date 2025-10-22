# streamlit_app.py
# Streamlit UI that shows ALL .md cases with a presort you can accept per row.
# Adapted to CSV schema: id, firstname, name, birthdate (German dd.mm.yyyy)
#
# Usage:
#   pip install streamlit rapidfuzz pandas tqdm
#   streamlit run streamlit_app.py
#
# Notes:
# - Local only; no data leaves your machine.
# - Base folder contains patient-id subfolders (destination). Load folder contains files to sort (source).
# - "Apply moves" performs filesystem operations and writes an audit CSV.
# - Includes an "Mark as Extra" option for low-signal files (e.g., empty letterheads).

import csv
import re
import shutil
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from rapidfuzz import fuzz
from tqdm import tqdm

# ----------------------------- Regex & Helpers -----------------------------
DATE_PATTERNS = [
    r"\b(20\d{2}|19\d{2})[-./](0?[1-9]|1[0-2])[-./](0?[1-9]|[12]\d|3[01])\b",  # YYYY-MM-DD
    r"\b(0?[1-9]|[12]\d|3[01])[-./](0?[1-9]|1[0-2])[-./](20\d{2}|19\d{2})\b",  # DD-MM-YYYY
    r"\b(0?[1-9]|1[0-2])[-./](0?[1-9]|[12]\d|3[01])[-./](20\d{2}|19\d{2})\b",  # MM-DD-YYYY
    r"\b([0-3]?\d)[.]([0-1]?\d)[.](19|20)\d{2}\b",  # DD.MM.YYYY (German)
]
RE_DATE = re.compile("|".join(f"({p})" for p in DATE_PATTERNS))
RE_NAME_2TOK = re.compile(r"\b([A-ZÄÖÜ][A-Za-zÄÖÜäöüß']+)\s+([A-ZÄÖÜ][A-Za-zÄÖÜäöüß']+)\b")
RE_DOB_DE = re.compile(r"\b([0-3]?\d)[.]([0-1]?\d)[.](19|20)\d{2}\b")

# Folder mapping for renaming
FOLDER_MAPPING = {
    "Arztbriefe Ambulanz bis 2020": "Arztbriefe",
    "Cytology": "Cytology", 
    "Flow cytology": "Flow-Cytology"
}

# Folders to ignore
IGNORED_FOLDERS = {"Guidelines"}


def norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").strip()


def fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower().strip()


def de_to_iso(d: str) -> Optional[str]:
    d = norm(d)
    m = RE_DOB_DE.fullmatch(d)
    if not m:
        d2 = d.replace("/", ".").replace("-", ".")
        m = RE_DOB_DE.fullmatch(d2)
        if not m:
            return None
    dd = int(m.group(1))
    mm = int(m.group(2))
    yyyy = int(d[-4:]) if len(d) >= 4 else None
    try:
        dt = datetime.strptime(f"{dd:02d}.{mm:02d}.{yyyy:04d}", "%d.%m.%Y")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def any_to_iso(d: str) -> Optional[str]:
    d = norm(d).replace("/", "-").replace(".", "-")
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y"):
        try:
            return datetime.strptime(d, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


# ----------------------------- Data Model -----------------------------
@dataclass
class Patient:
    pid: str
    firstname: str
    lastname: str
    fullname: str
    dob_de: str
    dob_iso: Optional[str]
    folded_full: str
    folded_first: str
    folded_last: str


def load_roster(csv_path: Path) -> Dict[str, Patient]:
    roster: Dict[str, Patient] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        
        # Strip spaces from column names to handle CSV formatting issues
        r.fieldnames = [name.strip() if name else name for name in (r.fieldnames or [])]
        
        required = {"id", "firstname", "name", "birthdate"}
        if not required.issubset({(h or '').strip().lower() for h in (r.fieldnames or [])}):
            raise ValueError(f"CSV must have columns: {required}")
        
        # Convert to list to get row count for progress bar
        rows = list(r)
        print(f"Loading {len(rows)} patients from roster...")
        
        for row in tqdm(rows, desc="Loading patient roster"):
            # Also strip spaces from the row values
            pid = norm(row.get("id", "").strip())
            first = norm(row.get("firstname", "").strip())
            last = norm(row.get("name", "").strip())
            dob_de = norm(row.get("birthdate", "").strip())
            dob_iso = de_to_iso(dob_de) if dob_de else None
            fullname = f"{first} {last}".strip()
            roster[pid] = Patient(
                pid=pid,
                firstname=first,
                lastname=last,
                fullname=fullname,
                dob_de=dob_de,
                dob_iso=dob_iso,
                folded_full=fold(fullname),
                folded_first=fold(first),
                folded_last=fold(last),
            )
    return roster


# ----------------------------- Signal Extraction -----------------------------

def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def extract_signals(p: Path, load_root: Path) -> dict:
    fn = p.name  # This is the filename
    text = read_text(p)
    # Include folder names in the search
    folder_path = str(p.parent.relative_to(load_root))  # Gets folder structure
    body = f"{folder_path}\n{fn}\n{text}"  # Now includes folder path
    names = set()
    for m in RE_NAME_2TOK.finditer("\n".join(body.splitlines()[:120])):
        names.add(f"{m.group(1)} {m.group(2)}")
        names.add(f"{m.group(2)} {m.group(1)}")
    dates = [m.group(0) for m in RE_DATE.finditer(body)]
    dobs_de = [m.group(0) for m in RE_DOB_DE.finditer(body)]
    # No longer extract IDs since they won't match
    return {"names": {norm(n) for n in names}, "dates": dates, "dobs_de": dobs_de}


# ----------------------------- Scoring -----------------------------

def score_candidate(sig: dict, pat: Patient) -> float:
    score = 0.0
    # Remove ID matching since patient IDs are new and won't appear in reports
    
    # DOB matching
    found_iso = {x for x in (any_to_iso(d) for d in sig["dobs_de"]) if x}
    if pat.dob_iso and pat.dob_iso in found_iso:
        score += 1.0  # Increased weight since no ID matching
    
    # Name matching
    folded_names = {fold(n) for n in sig["names"]}
    if pat.folded_full in folded_names:
        score += 0.8  # Increased weight
    elif pat.folded_last and any(pat.folded_last in n for n in folded_names):
        score += 0.5  # Increased weight
        best = 0
        for n in sig["names"]:
            for token in n.split():
                best = max(best, fuzz.ratio(fold(token), pat.folded_first))
        if best >= 90:
            score += 0.3  # Increased weight
    
    return min(score, 1.5)


# ----------------------------- Utilities -----------------------------

def should_process_file(p: Path, load_root: Path) -> bool:
    """Check if file should be processed based on folder structure"""
    try:
        rel = p.relative_to(load_root)
    except ValueError:
        return False
    
    if not rel.parts:
        return False
    
    top_folder = rel.parts[0]
    return top_folder not in IGNORED_FOLDERS


def get_folder_prefix(p: Path, load_root: Path) -> str:
    """Get folder prefix for file renaming"""
    try:
        rel = p.relative_to(load_root)
    except ValueError:
        return "Unknown"
    
    if not rel.parts:
        return "Unknown"
    
    top_folder = rel.parts[0]
    return FOLDER_MAPPING.get(top_folder, top_folder)


def pick_best_date(sig_dates: List[str], fallback_dt: datetime) -> str:
    iso = [any_to_iso(d.replace(".", "-")) for d in sig_dates]
    iso = [d for d in iso if d]
    if iso:
        iso.sort(reverse=True)
        return iso[0]
    return fallback_dt.strftime("%Y-%m-%d")


def unique_path(target: Path) -> Path:
    if not target.exists():
        return target
    stem, suf = target.stem, target.suffix
    for i in range(2, 100000):
        cand = target.with_name(f"{stem}-{i}{suf}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"Could not create unique path for {target}")


# ----------------------------- Streamlit UI -----------------------------
st.set_page_config(page_title="Patient File Sorter", layout="wide")
st.title("📁 Patient File Sorter — Name-Based Matching")

with st.sidebar:
    st.header("Settings")
    base_folder_str = st.text_input("Base folder (contains patient-id subfolders - DESTINATION)", value="")
    load_folder_str = st.text_input("Load folder (contains files to sort - SOURCE)", value="")
    csv_str = st.text_input("Roster CSV path (id,firstname,name,birthdate)", value="")
    auto_threshold = st.slider("Auto-accept presort with score ≥", 0.0, 1.5, 0.8, 0.05)  # Lowered default
    show_only_pending = st.checkbox("Show only pending (not accepted)", value=True)
    extra_folder_name = st.text_input("Extra bucket folder (under base)", value="_extra")
    
    st.subheader("📊 Processing Info")
    st.info("**Processed folders:**\n- Arztbriefe Ambulanz bis 2020\n- Cytology\n- Flow cytology\n\n**Ignored:**\n- Guidelines")

if not base_folder_str or not load_folder_str or not csv_str:
    st.info("Enter base folder, load folder, and roster CSV in the sidebar to begin.")
    st.stop()

base_folder = Path(base_folder_str).expanduser().resolve()
load_folder = Path(load_folder_str).expanduser().resolve()
csv_path = Path(csv_str).expanduser().resolve()

if not base_folder.exists():
    st.error(f"Base folder does not exist: {base_folder}")
    st.stop()
if not load_folder.exists():
    st.error(f"Load folder does not exist: {load_folder}")
    st.stop()
if not csv_path.exists():
    st.error(f"CSV not found: {csv_path}")
    st.stop()

# Quick CSV debug before processing everything
st.sidebar.write("**Quick CSV Check:**")
try:
    # Check first few lines of raw CSV
    with csv_path.open('r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        second_line = f.readline().strip()
    
    st.sidebar.write(f"Header: `{first_line}`")
    st.sidebar.write(f"First row: `{second_line}`")
    
    # Quick pandas check
    sample_df = pd.read_csv(csv_path, nrows=2)
    st.sidebar.write("Columns found:")
    for col in sample_df.columns:
        st.sidebar.write(f"- `{col}`")
    
except Exception as e:
    st.sidebar.error(f"CSV error: {e}")
    st.stop()

@st.cache_data(show_spinner=True)
def load_all(base_folder: Path, load_folder: Path, csv_path: Path):
    roster = load_roster(csv_path)
    
    # Debug: Check roster loading immediately
    print(f"Debug: Loaded {len(roster)} patients from CSV")
    print("First 3 patients:")
    for i, (pid, patient) in enumerate(list(roster.items())[:3]):
        print(f"  {i+1}. ID: '{pid}' | First: '{patient.firstname}' | Last: '{patient.lastname}' | Full: '{patient.fullname}' | DOB: '{patient.dob_de}'")
    
    # Debug: Check CSV raw data
    print("\nDebug: Raw CSV sample:")
    try:
        with csv_path.open('r', encoding='utf-8') as f:
            lines = [f.readline().strip() for _ in range(3)]
            for i, line in enumerate(lines):
                print(f"  Line {i+1}: {line}")
    except Exception as e:
        print(f"  Error reading raw CSV: {e}")
    
    # Find files in the load folder, excluding Guidelines
    print("Scanning for markdown files...")
    all_files = [p for p in tqdm(load_folder.rglob("*.md"), desc="Finding .md files") if p.is_file()]
    files = [p for p in tqdm(all_files, desc="Filtering processable files") if should_process_file(p, load_folder)]
    
    # Map patient IDs to directories in the base folder
    print("Mapping patient directories...")
    pid2dir = {}
    for d in tqdm(base_folder.iterdir(), desc="Scanning patient directories"):
        if d.is_dir():
            pid2dir[d.name] = d

    print(f"Processing {len(files)} files for patient matching...")
    rows = []
    for p in tqdm(files, desc="Processing files"):  # This should use 'files' not 'all_files'
        sig = extract_signals(p, load_folder)
        file_dt = datetime.fromtimestamp(p.stat().st_mtime)
        folder_prefix = get_folder_prefix(p, load_folder)
        best_date = pick_best_date(sig["dates"], file_dt)
        
        scored = []
        for pat in roster.values():
            s = score_candidate(sig, pat)
            if s > 0:
                scored.append((s, pat))
        scored.sort(key=lambda x: x[0], reverse=True)
        
        top_pid = scored[0][1].pid if scored else ""
        top_name = scored[0][1].fullname if scored else ""
        top_dob = scored[0][1].dob_de if scored else ""
        top_score = float(scored[0][0]) if scored else 0.0

        rows.append(dict(
            source=str(p),
            rel=str(p.relative_to(load_folder)),
            preview=read_text(p)[:800],
            folder_prefix=folder_prefix,
            suggested_date=best_date,
            names_found="|".join(sorted(sig["names"])),
            dobs_found="|".join(sorted(sig["dobs_de"])),
            top_pid=top_pid,
            top_name=top_name,
            top_dob=top_dob,
            top_score=round(top_score, 3),
            is_signal_poor=(len(sig["names"]) == 0 and len(sig["dobs_de"]) == 0),
        ))

    df = pd.DataFrame(rows)
    print(f"Processing complete! Found {len(df)} files to process.")
    return roster, pid2dir, df, len(all_files), len(files)

roster, pid2dir, df, total_files, processed_files = load_all(base_folder, load_folder, csv_path)

# Add score distribution summary
print("\n" + "="*50)
print("SCORE DISTRIBUTION SUMMARY")
print("="*50)

# Define score buckets
score_buckets = [
    (1.5, float('inf'), "Perfect (1.5)"),
    (1.0, 1.5, "Excellent (1.0-1.4)"),
    (0.8, 1.0, "Good (0.8-0.9)"),
    (0.5, 0.8, "Fair (0.5-0.7)"),
    (0.1, 0.5, "Poor (0.1-0.4)"),
    (0.0, 0.1, "No match (0.0)")
]

total_files_processed = len(df)
for min_score, max_score, label in score_buckets:
    if max_score == float('inf'):
        count = len(df[df['top_score'] >= min_score])
    else:
        count = len(df[(df['top_score'] >= min_score) & (df['top_score'] < max_score)])
    
    percentage = (count / total_files_processed * 100) if total_files_processed > 0 else 0
    print(f"{label:20} | {count:4d} files ({percentage:5.1f}%)")

print("-" * 50)
print(f"{'Total':20} | {total_files_processed:4d} files (100.0%)")

# Show auto-accept threshold info
auto_accept_count = len(df[df['top_score'] >= auto_threshold])
auto_accept_pct = (auto_accept_count / total_files_processed * 100) if total_files_processed > 0 else 0
print(f"\nFiles above auto-accept threshold ({auto_threshold}): {auto_accept_count} ({auto_accept_pct:.1f}%)")
print("="*50 + "\n")

# Session state
if "assign" not in st.session_state:
    st.session_state.assign = {}
if "accepted" not in st.session_state:
    st.session_state.accepted = set()
if "extras" not in st.session_state:
    st.session_state.extras = set()

# Display folder info
st.info(f"📂 **Loading from:** `{load_folder}`\n\n📁 **Moving to:** `{base_folder}`\n\n📄 **Files found:** {processed_files} processable / {total_files} total\n\n👥 **Patients in roster:** {len(roster)}")

# Filters
colf1, colf2, colf3, colf4, colf5 = st.columns([2, 2, 1, 1, 1])
with colf1:
    q = st.text_input("Filter path/content/names", "")
with colf2:
    name_filter = st.text_input("Filter suggested name", "")
with colf3:
    min_score = st.number_input("Min score", 0.0, 1.5, 0.0, 0.05)
with colf4:
    max_rows = st.number_input("Max rows shown", 10, 5000, 300, 10)
with colf5:
    sort_by = st.selectbox("Sort by", ["Score (high first)", "Score (low first)", "Filename"], index=0)

mask = (df["top_score"] >= min_score)
if q:
    ql = q.lower()
    mask &= (
        df["rel"].str.lower().str.contains(ql) |
        df["source"].str.lower().str.contains(ql) |
        df["names_found"].str.lower().str.contains(ql) |
        df["preview"].str.lower().str.contains(ql)
    )
if name_filter:
    nl = name_filter.lower()
    mask &= df["top_name"].str.lower().str.contains(nl)
if show_only_pending:
    mask &= ~df["source"].isin(list(st.session_state.accepted) + list(st.session_state.extras))

view = df[mask].copy().head(int(max_rows))

# Sort based on user selection
if sort_by == "Score (high first)":
    view = view.sort_values('top_score', ascending=False)
elif sort_by == "Score (low first)":
    view = view.sort_values('top_score', ascending=True)
elif sort_by == "Filename":
    view = view.sort_values('rel', ascending=True)

# Initialize presel for high-confidence rows
for _, r in view.iterrows():
    src = r["source"]
    if src not in st.session_state.assign:
        if r["top_score"] >= auto_threshold and r["top_pid"]:
            st.session_state.assign[src] = r["top_pid"]
        else:
            st.session_state.assign[src] = ""

# Roster options
pid_options = [""] + list(roster.keys())  # Add empty option at the start
label_by_pid = {"": "-- Select Patient --"}  # Add label for empty option
label_by_pid.update({p.pid: f"{p.fullname} ({p.dob_de}) — ID: {p.pid}" for p in roster.values()})

st.subheader(f"Cases ({len(view)} shown / {len(df)} total)")

for _, r in view.iterrows():
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([3, 3, 2, 2])
        src = r["source"]
        rel = r["rel"]
        st.caption(rel)
        with c1:
            st.markdown(f"**Suggested:** {r['top_name']}  ")
            st.markdown(f"DOB (suggested): `{r['top_dob']}`  ")
            st.markdown(f"Score: `{r['top_score']}`  ")
            st.markdown(f"Names: `{r['names_found']}`  ")
            st.markdown(f"DOBs found: `{r['dobs_found']}`  ")
            st.markdown(f"Folder: `{r['folder_prefix']}`")
            if r["is_signal_poor"]:
                st.warning("Low-signal file (likely extra type)")
        with c2:
            with st.expander("Preview (first ~800 chars"):
                st.code(read_text(Path(src))[:800] or "<empty>", language="markdown")
        with c3:
            default_pid = st.session_state.assign.get(src, "")
            try:
                idx_default = pid_options.index(default_pid) if default_pid in pid_options else 0
            except ValueError:
                idx_default = 0
            sel = st.selectbox(
                "Assign patient",
                options=pid_options,
                index=idx_default,
                format_func=lambda pid: label_by_pid.get(pid, pid),
                key=f"sel_{src}",
            )
            st.session_state.assign[src] = sel
        with c4:
            acc_btn = st.button("✅ Accept", key=f"accept_{src}")
            extra_btn = st.button("🗂️ Mark as Extra", key=f"extra_{src}")
            if acc_btn:
                st.session_state.accepted.add(src)
                st.success("Accepted; will be applied on 'Apply moves'")
            if extra_btn:
                st.session_state.extras.add(src)
                st.success("Marked as Extra; will move to extra bucket on 'Apply moves'")
            
            chosen_pid = st.session_state.assign.get(src, "")
            chosen_label = label_by_pid.get(chosen_pid, "")
            # Show original filename instead of generated name
            new_fname = Path(src).name  # Keep original name
            st.caption(f"New name → **{new_fname}**  |  Target → {chosen_label or '<choose>'}")

st.divider()
colA, colB, colC, colD = st.columns([2, 2, 2, 3])
with colA:
    accept_auto = st.button("Accept all with score ≥ threshold")
    if accept_auto:
        auto = df[(df.top_score >= auto_threshold) & (df.top_pid != "")]
        for _, rr in auto.iterrows():
            st.session_state.assign[rr["source"]] = rr["top_pid"]
            st.session_state.accepted.add(rr["source"])
        st.success(f"Accepted {len(auto)} auto-suggested cases")
with colB:
    clear_acc = st.button("Clear accepted (session)")
    if clear_acc:
        st.session_state.accepted = set()
        st.success("Cleared accepted set")
with colC:
    clear_extra = st.button("Clear extras (session)")
    if clear_extra:
        st.session_state.extras = set()
        st.success("Cleared extras set")
with colD:
    if st.button("🚀 Fully Auto Process (score ≥ threshold)"):
        auto_files = df[(df.top_score >= auto_threshold) & (df.top_pid != "")]
        low_files = df[df.top_score < auto_threshold]
        
        # Auto-accept high confidence
        for _, rr in auto_files.iterrows():
            st.session_state.assign[rr["source"]] = rr["top_pid"]
            st.session_state.accepted.add(rr["source"])
        
        # Auto-mark low confidence as extras
        for _, rr in low_files.iterrows():
            st.session_state.extras.add(rr["source"])
            
        st.success(f"Auto-processed: {len(auto_files)} accepted, {len(low_files)} marked as extra")

apply = st.button("🚚 Apply moves and renames (writes to disk)")
if apply:
    applied = 0
    extra_applied = 0
    log_rows = []

    # No need to create extra directory since we're ignoring extras
    # extra_dir = (base_folder / extra_folder_name)
    # extra_dir.mkdir(parents=True, exist_ok=True)

    accepted_list = list(st.session_state.accepted)
    if accepted_list:
        print(f"Processing {len(accepted_list)} accepted files...")
        for src in tqdm(accepted_list, desc="Moving accepted files"):
            pid = st.session_state.assign.get(src, "")
            if not pid:
                continue
            p = Path(src)
            if not p.exists():
                st.warning(f"Missing file: {src}")
                continue
            
            # Target directory is in the base folder
            target_dir = pid2dir.get(pid)
            if target_dir is None:
                target_dir = base_folder / pid
                target_dir.mkdir(parents=True, exist_ok=True)
                pid2dir[pid] = target_dir
            
            sig = extract_signals(p, load_folder)
            file_dt = datetime.fromtimestamp(p.stat().st_mtime)
            folder_prefix = get_folder_prefix(p, load_folder)
            best_date = pick_best_date(sig["dates"], file_dt)
            # Keep original filename instead of renaming
            new_name = p.name  # Use original filename
            target = unique_path(target_dir / new_name)
            shutil.copy2(p, target)
            applied += 1
            log_rows.append(dict(
                source=src,
                action="move+rename",
                patient_id=pid,
                new_path=str(target),
                date=best_date,
                folder_prefix=folder_prefix,
            ))

    # Log ignored extras instead of moving them
    extras_list = list(st.session_state.extras)
    if extras_list:
        print(f"Ignoring {len(extras_list)} extra files...")
        for src in tqdm(extras_list, desc="Logging ignored extra files"):
            p = Path(src)
            if not p.exists():
                st.warning(f"Missing file: {src}")
                continue
            sig = extract_signals(p, load_folder)
            file_dt = datetime.fromtimestamp(p.stat().st_mtime)
            folder_prefix = get_folder_prefix(p, load_folder)
            best_date = pick_best_date(sig["dates"], file_dt)
            extra_applied += 1
            log_rows.append(dict(
                source=src,
                action="ignored-extra",
                patient_id="",
                new_path="",  # No new path since we're not moving
                date=best_date,
                folder_prefix=folder_prefix,
            ))

    if log_rows:
        print("Writing audit log...")
        log_name = f"file_sort_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        out_path = (base_folder / log_name).resolve()
        pd.DataFrame(log_rows).to_csv(out_path, index=False)
        st.success(f"Applied {applied} patient moves and ignored {extra_applied} extras. Audit log: {out_path}")
        print(f"Complete! Audit log saved to: {out_path}")
    else:
        st.info("Nothing to apply. Accept or mark files first.")
