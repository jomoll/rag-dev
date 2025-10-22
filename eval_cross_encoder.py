#!/usr/bin/env python3
# eval_reranker.py
import argparse, math, os, re, sqlite3, json
from typing import List, Dict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------- SQLite I/O --------
def connect(db_path: str) -> sqlite3.Connection:
    qa_db_path = db_path.replace(".sqlite", "_qa.sqlite")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    if os.path.exists(qa_db_path):
        conn.execute(f"ATTACH DATABASE '{qa_db_path}' AS qa_db")
    return conn

def load_corpus(conn):
    rows = conn.execute("""
      SELECT rs.section_id, rs.report_id, UPPER(rs.name) AS section_name,
             rs.text AS text, r.patient_id
      FROM report_sections rs
      JOIN reports r ON r.report_id = rs.report_id
      WHERE rs.text IS NOT NULL AND LENGTH(rs.text) > 0
    """).fetchall()
    sec_text, sec_report, sec_patient, sec_section_name = {}, {}, {}, {}
    for r in rows:
        sid = int(r["section_id"])
        sec_text[sid] = r["text"]
        sec_report[sid] = r["report_id"]
        sec_patient[sid] = r["patient_id"]
        sec_section_name[sid] = r["section_name"] or "UNKNOWN"
    return sec_text, sec_report, sec_patient, sec_section_name

def load_qa(conn) -> List[dict]:
    try:
        rows = conn.execute("""
          SELECT qi.qa_id, qi.question, qi.section_id AS gold_section_id,
                 qi.report_id AS gold_report_id,
                 UPPER(qi.section_name) AS gold_section_name,
                 r.patient_id AS patient_id
          FROM qa_db.qa_items qi
          JOIN main.reports r ON r.report_id = qi.report_id
        """).fetchall()
    except:
        rows = conn.execute("""
          SELECT qi.qa_id, qi.question, qi.section_id AS gold_section_id,
                 qi.report_id AS gold_report_id,
                 UPPER(qi.section_name) AS gold_section_name,
                 r.patient_id AS patient_id
          FROM qa_items qi
          JOIN reports r ON r.report_id = qi.report_id
        """).fetchall()
    out = []
    for r in rows:
        if r["gold_section_id"] is None: 
            continue
        out.append({
            "qa_id": int(r["qa_id"]),
            "question": r["question"],
            "gold_section_id": int(r["gold_section_id"]),
            "gold_report_id": r["gold_report_id"],
            "gold_section_name": (r["gold_section_name"] or "UNKNOWN"),
            "patient_id": r["patient_id"],
        })
    return out

# -------- Tiny BM25 (char-3gram) --------
def _normalize_text(s: str) -> str: return (s or "").lower()
def _char_ngrams(s: str, n: int = 3):
    s = re.sub(r"[^0-9a-zA-ZäöüÄÖÜß]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().replace(" ", "_")
    if len(s) < n: return [s] if s else []
    return [s[i:i+n] for i in range(len(s)-n+1)]

class BM25:
    def __init__(self, k1=1.2, b=0.75, ngram=3):
        self.k1, self.b, self.ngram = k1, b, ngram
        self.df, self.post, self.doc_len = {}, {}, []
        self.N, self.avgdl = 0, 0.0
    def add(self, docs: List[str]):
        self.N = len(docs); self.df.clear(); self.post.clear(); self.doc_len=[]
        for i, t in enumerate(docs):
            tf = {}
            for tok in _char_ngrams(_normalize_text(t), self.ngram):
                tf[tok] = tf.get(tok, 0) + 1
            self.doc_len.append(sum(tf.values()))
            for tok, c in tf.items():
                self.df[tok] = self.df.get(tok, 0) + 1
                self.post.setdefault(tok, {})[i] = c
        self.avgdl = (sum(self.doc_len)/max(1,self.N)) if self.N else 0.0
    def score(self, query: str, cand_idx: np.ndarray) -> np.ndarray:
        toks = set(_char_ngrams(_normalize_text(query), self.ngram))
        idf = {t: math.log((self.N - self.df.get(t,0) + .5) / (self.df.get(t,0) + .5) + 1.0) for t in toks}
        scores = np.zeros(len(cand_idx), dtype=np.float32)
        for j, di in enumerate(cand_idx):
            dl = self.doc_len[di] if di < len(self.doc_len) else 0
            norm = self.k1 * (1 - self.b + self.b * (dl / self.avgdl)) if self.avgdl>0 else self.k1
            s=0.0
            for t in toks:
                tf = self.post.get(t, {}).get(di, 0)
                if tf==0: continue
                s += idf[t] * ((tf*(self.k1+1.0)) / (tf + norm))
            scores[j]=s
        return scores

# -------- Metrics --------
def recall_at_k(ranks: List[int], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r < k) / max(1, len(ranks))
def mrr(ranks: List[int]) -> float:
    return sum(1.0/(r+1) for r in ranks if r is not None) / max(1, len(ranks))
def ndcg_at_k(rel_lists: List[List[int]], k: int) -> float:
    total=0.0
    for rel in rel_lists:
        gains=[rel[i]/math.log2(i+2) for i in range(min(k,len(rel)))]
        dcg=sum(gains)
        ideal=sorted(rel, reverse=True)
        idcg=sum(ideal[i]/math.log2(i+2) for i in range(min(k,len(ideal))))
        total += (dcg/idcg) if idcg>0 else 0.0
    return total/max(1,len(rel_lists))

# -------- Eval (BM25 -> CE rerank) --------
@torch.no_grad()
def evaluate(model_name: str, db_path: str, bm25_topk: int, restrict_same_patient: bool,
             restrict_same_report: bool, max_len: int, batch_size: int, device: str):
    conn = connect(db_path)
    try:
        qa = load_qa(conn)
        sec_text, sec_report, sec_patient, _ = load_corpus(conn)
    finally:
        conn.close()

    # Build arrays for BM25
    section_ids = sorted(sec_text.keys())
    sid2row = {sid:i for i,sid in enumerate(section_ids)}
    docs = [sec_text[sid] for sid in section_ids]
    doc_pat = np.array([sec_patient[sid] for sid in section_ids], dtype=object)
    doc_rep = np.array([sec_report[sid] for sid in section_ids], dtype=object)

    bm25 = BM25(); bm25.add(docs)

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device); model.eval()

    Kmax = 20
    ranks, rel_lists = [], []

    for q in qa:
        gold_row = sid2row.get(q["gold_section_id"])
        if gold_row is None:
            ranks.append(None); rel_lists.append([0]*Kmax); continue

        mask = np.ones(len(section_ids), dtype=bool)
        if restrict_same_patient and q.get("patient_id"):
            mask &= (doc_pat == q["patient_id"])
        if restrict_same_report and q.get("gold_report_id"):
            mask &= (doc_rep == q["gold_report_id"])
        cand_idx = np.nonzero(mask)[0]
        if len(cand_idx)==0:
            ranks.append(None); rel_lists.append([0]*Kmax); continue

        # BM25 candidate set
        scores = bm25.score(q["question"], cand_idx)
        order = np.argsort(-scores)
        cand_idx = cand_idx[order[:bm25_topk]]

        # Cross-encoder scores
        q_texts = [q["question"]]*len(cand_idx)
        d_texts = [sec_text[section_ids[i]] for i in cand_idx]
        ce_scores=[]
        for i in range(0, len(cand_idx), batch_size):
            enc = tok(q_texts[i:i+batch_size], d_texts[i:i+batch_size],
                      truncation=True, padding=True, max_length=max_len, return_tensors="pt").to(device)
            logits = model(**enc).logits.view(-1)
            ce_scores.append(logits.detach().cpu().numpy())
        ce_scores = np.concatenate(ce_scores, axis=0) if ce_scores else np.array([])

        rerank = np.argsort(-ce_scores)
        top_rows_global = [sid2row[section_ids[cand_idx[j]]] for j in rerank[:Kmax]]

        rr=None; rel=[]
        for rank, r in enumerate(top_rows_global):
            is_rel = 1 if r == gold_row else 0
            rel.append(is_rel)
            if rr is None and is_rel: rr = rank
        if len(rel)<Kmax: rel += [0]*(Kmax-len(rel))
        ranks.append(rr); rel_lists.append(rel[:Kmax])

    metrics = {f"Recall@{k}": recall_at_k(ranks,k) for k in [1,5,10,20]}
    metrics.update({f"nDCG@{k}": ndcg_at_k(rel_lists,k) for k in [1,5,10,20]})
    metrics["MRR"] = mrr(ranks)
    print("\n=== Cross-Encoder Rerank Evaluation ===")
    for k in [1,5,10,20]:
        print(f"Recall@{k}: {metrics[f'Recall@{k}']:.3f}   nDCG@{k}: {metrics[f'nDCG@{k}']:.3f}")
    print(f"MRR: {metrics['MRR']:.3f}")
    return metrics

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--model", required=True, help="Path or HF id of cross-encoder (e.g., BAAI/bge-reranker-v2-m3 or your fine-tuned dir)")
    ap.add_argument("--bm25-topk", type=int, default=100)
    ap.add_argument("--restrict-same-patient", type=int, default=1)
    ap.add_argument("--restrict-same-report", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-json", default="")
    args = ap.parse_args()

    m = evaluate(
        model_name=args.model, db_path=args.db, bm25_topk=args.bm25_topk,
        restrict_same_patient=bool(args.restrict_same_patient),
        restrict_same_report=bool(args.restrict_same_report),
        max_len=args.max_len, batch_size=args.batch_size, device=args.device
    )
    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)
