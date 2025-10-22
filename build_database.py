#!/usr/bin/env python3
"""
Build SQLite database from patient .md files with metadata extraction.
Extracts patient name from CSV, date from filename, and report type from parentheses.
python build_database.py /data/moll/interagt/processed/combined patients.csv database/base_database.sqlite
"""

import os
import sqlite3
import hashlib
import datetime
import re
import csv
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm

def sha256_of_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def init_db(conn: sqlite3.Connection):
    """Initialize database schema"""
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA cache_size = 20000")
    
    conn.executescript("""
    PRAGMA foreign_keys=ON;
    
    CREATE TABLE IF NOT EXISTS patients(
        patient_id TEXT PRIMARY KEY,
        firstname TEXT,
        lastname TEXT,
        fullname TEXT,
        dob TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS reports(
        report_id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL,
        filename TEXT,
        report_type TEXT,
        report_date TEXT,
        created_at TEXT,
        source_path TEXT,
        sha256 TEXT UNIQUE,
        content TEXT,
        FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
    );
    
    CREATE VIRTUAL TABLE IF NOT EXISTS report_fts USING fts5(
        report_id UNINDEXED, 
        patient_id UNINDEXED, 
        filename UNINDEXED,
        report_type UNINDEXED,
        content,
        tokenize='porter'
    );
    
    CREATE INDEX IF NOT EXISTS idx_reports_patient_date ON reports(patient_id, report_date);
    CREATE INDEX IF NOT EXISTS idx_reports_type ON reports(report_type);
    """)
    conn.commit()

def load_patient_roster(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """Load patient information from CSV"""
    roster = {}
    
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Strip spaces from column names
        reader.fieldnames = [name.strip() if name else name for name in (reader.fieldnames or [])]
        
        for row in reader:
            pid = row.get('id', '').strip()
            firstname = row.get('firstname', '').strip()
            lastname = row.get('name', '').strip()
            dob = row.get('birthdate', '').strip()
            
            roster[pid] = {
                'firstname': firstname,
                'lastname': lastname,
                'fullname': f"{firstname} {lastname}".strip(),
                'dob': dob
            }
    
    return roster

def extract_date_from_filename(filename: str) -> Optional[str]:
    """Extract date from filename in various formats"""
    # Patterns for dd.mm.yyyy and dd.mm.yy
    patterns = [
        r'\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b',  # dd.mm.yyyy
        r'\b(\d{1,2})\.(\d{1,2})\.(\d{2})\b',  # dd.mm.yy
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            day, month, year = match.groups()
            
            # Convert 2-digit year to 4-digit
            if len(year) == 2:
                year_int = int(year)
                # Assume years 00-30 are 2000s, 31-99 are 1900s
                if year_int <= 30:
                    year = f"20{year}"
                else:
                    year = f"19{year}"
            
            # Validate and format
            try:
                dt = datetime.datetime.strptime(f"{day}.{month}.{year}", "%d.%m.%Y")
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
    
    return None

def extract_report_type_from_filename(filename: str) -> str:
    """Extract report type from parentheses in filename"""
    # Look for text in parentheses
    match = re.search(r'\(([^)]+)\)', filename)
    if match:
        content = match.group(1).strip()
        # Only accept specific report types
        if content in ["Arztbrief", "Cytologie", "Flow cytometry"]:
            return content
    
    # Default to "Arztbrief" for everything else
    return "Arztbrief"

def safe_report_id(filename: str, patient_id: str) -> str:
    """Generate safe report ID"""
    stem = os.path.splitext(filename)[0]
    base = re.sub(r"\s+", "_", stem).strip("_")
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)[:60]
    scoped_id = f"{patient_id}_{base}"
    return scoped_id or f"{patient_id}_{sha256_of_text(stem)[:12]}"

def upsert_patient(conn: sqlite3.Connection, patient_id: str, patient_info: Dict[str, str]):
    """Insert or update patient information"""
    conn.execute("""
        INSERT OR REPLACE INTO patients(patient_id, firstname, lastname, fullname, dob)
        VALUES (?, ?, ?, ?, ?)
    """, (
        patient_id,
        patient_info.get('firstname', ''),
        patient_info.get('lastname', ''),
        patient_info.get('fullname', ''),
        patient_info.get('dob', '')
    ))

def insert_report(conn: sqlite3.Connection, patient_id: str, report_id: str, filename: str, 
                 content: str, source_path: str, report_type: str, report_date: Optional[str]):
    """Insert report into database"""
    
    # Check for duplicate content (true duplicates)
    content_hash = sha256_of_text(content)
    duplicate = conn.execute(
        "SELECT report_id FROM reports WHERE patient_id=? AND sha256=?", 
        (patient_id, content_hash)
    ).fetchone()
    
    if duplicate:
        print(f"[SKIP] Duplicate content for patient {patient_id}: {filename}")
        return False, True  # Return (success, was_duplicate)
    
    # Handle report_id conflicts by making unique
    original_report_id = report_id
    counter = 1
    while True:
        existing = conn.execute("SELECT 1 FROM reports WHERE report_id=?", (report_id,)).fetchone()
        if not existing:
            break
        # Add counter to make unique
        report_id = f"{original_report_id}_{counter}"
        counter += 1
    
    if report_id != original_report_id:
        print(f"[INFO] Report ID conflict resolved: {original_report_id} -> {report_id}")
    
    created_at = datetime.datetime.now().isoformat(timespec="seconds")
    
    try:
        conn.execute("""
            INSERT INTO reports(
                report_id, patient_id, filename, report_type, report_date, 
                created_at, source_path, sha256, content
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            report_id, patient_id, filename, report_type, report_date,
            created_at, str(source_path), content_hash, content
        ))
        
        # Add to FTS index
        conn.execute("""
            INSERT INTO report_fts(report_id, patient_id, filename, report_type, content)
            VALUES (?, ?, ?, ?, ?)
        """, (report_id, patient_id, filename, report_type, content))
        
        return True, False  # Return (success, was_duplicate)
        
    except sqlite3.IntegrityError as e:
        print(f"[ERROR] Failed to insert report {report_id}: {e}")
        return False, False  # Return (success, was_duplicate)

def process_patient_directory(conn: sqlite3.Connection, patient_dir: Path, 
                            patient_roster: Dict[str, Dict[str, str]]):
    """Process all .md files in a patient directory"""
    patient_id = patient_dir.name
    patient_info = patient_roster.get(patient_id, {
        'firstname': '', 'lastname': '', 'fullname': '', 'dob': ''
    })
    
    # Upsert patient info
    upsert_patient(conn, patient_id, patient_info)
    
    processed_count = 0
    duplicate_count = 0
    
    # Process all .md files
    for md_file in patient_dir.glob("*.md"):
        try:
            # Read file content
            content = md_file.read_text(encoding='utf-8', errors='ignore').strip()
            if not content:
                continue
            
            # Extract metadata
            filename = md_file.name
            report_type = extract_report_type_from_filename(filename)
            report_date = extract_date_from_filename(filename)
            report_id = safe_report_id(filename, patient_id)
            
            # Insert report
            success, was_duplicate = insert_report(conn, patient_id, report_id, filename, content, 
                                                 str(md_file), report_type, report_date)
            if success:
                processed_count += 1
            elif was_duplicate:
                duplicate_count += 1
                
        except Exception as e:
            print(f"[ERROR] Processing {md_file}: {e}")
            continue
    
    return processed_count, duplicate_count

def build_database(base_folder: str, csv_path: str, db_path: str):
    """Main function to build the database"""
    
    base_path = Path(base_folder)
    csv_file = Path(csv_path)
    
    if not base_path.exists():
        raise FileNotFoundError(f"Base folder not found: {base_path}")
    
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    # Load patient roster
    print("Loading patient roster...")
    patient_roster = load_patient_roster(csv_file)
    print(f"Loaded {len(patient_roster)} patients from roster")
    
    # Initialize database
    print(f"Initializing database: {db_path}")
    conn = sqlite3.connect(db_path, timeout=30.0)
    init_db(conn)
    
    # Get all patient directories
    patient_dirs = [d for d in base_path.iterdir() if d.is_dir() and not d.name.startswith('_')]
    print(f"Found {len(patient_dirs)} patient directories")
    
    total_reports = 0
    total_duplicates = 0
    
    # Process each patient directory
    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        try:
            processed_count, duplicate_count = process_patient_directory(conn, patient_dir, patient_roster)
            total_reports += processed_count
            total_duplicates += duplicate_count
            
            # Commit every 100 patients
            if len([d for d in patient_dirs if d <= patient_dir]) % 100 == 0:
                conn.commit()
                
        except Exception as e:
            print(f"[ERROR] Processing directory {patient_dir}: {e}")
            continue
    
    # Final commit
    conn.commit()
    
    # Print summary
    patient_count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    report_count = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
    
    print(f"\n=== DATABASE SUMMARY ===")
    print(f"Patients: {patient_count}")
    print(f"Reports: {report_count}")
    print(f"Duplicates skipped: {total_duplicates}")
    print(f"Total files processed: {total_reports + total_duplicates}")
    
    # Report type breakdown
    print(f"\nReport types:")
    type_counts = conn.execute("""
        SELECT report_type, COUNT(*) 
        FROM reports 
        GROUP BY report_type 
        ORDER BY COUNT(*) DESC
    """).fetchall()
    
    for report_type, count in type_counts:
        print(f"  {report_type}: {count}")
    
    conn.close()
    print(f"\nDatabase saved to: {db_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python build_database.py <base_folder> <csv_path> <db_path>")
        print("Example: python build_database.py /data/patients patients.csv reports.db")
        sys.exit(1)
    
    base_folder = sys.argv[1]
    csv_path = sys.argv[2] 
    db_path = sys.argv[3]
    
    build_database(base_folder, csv_path, db_path)

