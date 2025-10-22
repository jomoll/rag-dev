#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a multilingual cross-encoder reranker on German clinical QA.

- Positives: (question, gold snippet)
- Negatives: BM25-mined within-patient hard negatives (+ optional random)
- Loss: BCEWithLogits (binary relevance)
- Eval: Recall@{1,5,10,20}, nDCG@{1,5,10,20}, MRR via reranking BM25 top-K

Run (typical):
    python train_cross_encoder.py \
        --db myeloma_reports.sqlite \
        --base-model BAAI/bge-reranker-v2-m3 \
        --output-dir reranker-bge-m3-myelo \
        --per-device-train-batch-size 16 \
        --epochs 3 \
        --negatives-per-pos 4 \
        --bm25-train-topk 50 \
        --bm25-eval-topk 100 \
        --restrict-same-report 0

Notes
- Uses patient-held-out split (80/20).
- BM25 uses char-3grams (robust for German compounds).
"""

import argparse
import math
import os
import random
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# -----------------------------
# SQLite I/O
# -----------------------------

def connect(db_path: str) -> sqlite3.Connection:
    qa_db_path = db_path.replace(".sqlite", "_qa.sqlite")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    if os.path.exists(qa_db_path):
        conn.execute(f"ATTACH DATABASE '{qa_db_path}' AS qa_db")
    return conn

def load_corpus(conn: sqlite3.Connection):
    rows = conn.execute("""
      SELECT rs.section_id, rs.report_id, UPPER(rs.name) AS section_name,
             rs.text AS text, r.patient_id
      FROM report_sections rs
      JOIN reports r ON r.report_id = rs.report_id
      WHERE rs.text IS NOT NULL AND LENGTH(rs.text) > 0
    """).fetchall()
    if not rows:
        raise RuntimeError("No report_sections text found.")

    sec_text = {}
    sec_report = {}
    sec_section_name = {}
    sec_patient = {}

    for r in rows:
        sid = int(r["section_id"])
        sec_text[sid] = r["text"]
        sec_report[sid] = r["report_id"]
        sec_section_name[sid] = r["section_name"] or "UNKNOWN"
        sec_patient[sid] = r["patient_id"]

    return sec_text, sec_report, sec_section_name, sec_patient

def load_qa(conn: sqlite3.Connection) -> List[dict]:
    try:
        qas = conn.execute("""
          SELECT qi.qa_id, qi.question, qi.section_id AS gold_section_id,
                 qi.report_id AS gold_report_id,
                 UPPER(qi.section_name) AS gold_section_name,
                 COALESCE(qi.phenomena, '[]') AS phenomena_json,
                 r.patient_id AS patient_id
          FROM qa_db.qa_items qi
          JOIN main.reports r ON r.report_id = qi.report_id
        """).fetchall()
    except:
        qas = conn.execute("""
          SELECT qi.qa_id, qi.question, qi.section_id AS gold_section_id,
                 qi.report_id AS gold_report_id,
                 UPPER(qi.section_name) AS gold_section_name,
                 COALESCE(qi.phenomena, '[]') AS phenomena_json,
                 r.patient_id AS patient_id
          FROM qa_items qi
          JOIN reports r ON r.report_id = qi.report_id
        """).fetchall()

    out = []
    for r in qas:
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

# -----------------------------
# BM25 (char-3gram)
# -----------------------------

def _normalize_text(s: str) -> str:
    return (s or "").lower()

def _char_ngrams(s: str, n: int = 3) -> List[str]:
    s = re.sub(r"[^0-9a-zA-ZäöüÄÖÜß]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(" ", "_")
    if len(s) < n:
        return [s] if s else []
    return [s[i:i+n] for i in range(len(s)-n+1)]

class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75, ngram: int = 3):
        self.k1 = k1
        self.b = b
        self.ngram = ngram
        self.df: Dict[str, int] = {}
        self.postings: Dict[str, Dict[int, int]] = {}
        self.doc_len: List[int] = []
        self.avgdl = 0.0
        self.N = 0

    def add_docs(self, docs: List[str]):
        self.N = len(docs)
        self.df = {}
        self.postings = {}
        self.doc_len = []
        for doc_id, text in enumerate(docs):
            toks = _char_ngrams(_normalize_text(text), self.ngram)
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self.doc_len.append(sum(tf.values()))
            for t, c in tf.items():
                self.df[t] = self.df.get(t, 0) + 1
                if t not in self.postings:
                    self.postings[t] = {}
                self.postings[t][doc_id] = c
        self.avgdl = (sum(self.doc_len) / max(1, self.N)) if self.N else 0.0

    def score(self, query: str, cand_idx: np.ndarray) -> np.ndarray:
        toks = _char_ngrams(_normalize_text(query), self.ngram)
        uniq = set(toks)
        idf = {}
        for t in uniq:
            df_t = self.df.get(t, 0)
            idf[t] = math.log((self.N - df_t + 0.5) / (df_t + 0.5) + 1.0) if self.N else 0.0

        scores = np.zeros(len(cand_idx), dtype=np.float32)
        for j, di in enumerate(cand_idx):
            dl = self.doc_len[di] if di < len(self.doc_len) else 0
            norm = self.k1 * (1 - self.b + self.b * (dl / self.avgdl)) if self.avgdl > 0 else self.k1
            s = 0.0
            for t in uniq:
                tf = self.postings.get(t, {}).get(di, 0)
                if tf == 0:
                    continue
                num = tf * (self.k1 + 1.0)
                s += idf[t] * (num / (tf + norm))
            scores[j] = s
        return scores

# -----------------------------
# Pair construction
# -----------------------------

@dataclass
class Pair:
    q: str
    d: str
    label: float  # 1.0 positive, 0.0 negative

def build_pairs(
    qa: List[dict],
    sec_text: Dict[int, str],
    sec_report: Dict[int, str],
    sec_patient: Dict[int, str],
    negatives_per_pos: int = 4,
    bm25_topk: int = 50,
    restrict_same_report: bool = False,
    seed: int = 1234,
) -> Tuple[List[Pair], List[Pair]]:
    """
    Create train/dev pairs with patient-held-out split.
    Dev uses 20% of patients.
    Negatives are BM25-mined within patient (and optionally report).
    """
    rng = random.Random(seed)

    # Build arrays aligned for BM25 indexing
    all_section_ids = sorted(sec_text.keys())
    sid_to_row = {sid: i for i, sid in enumerate(all_section_ids)}
    docs = [sec_text[sid] for sid in all_section_ids]
    doc_pat = np.array([sec_patient[sid] for sid in all_section_ids], dtype=object)
    doc_rep = np.array([sec_report[sid] for sid in all_section_ids], dtype=object)

    bm25 = BM25Index()
    bm25.add_docs(docs)

    # Split by patient
    patients = sorted({q["patient_id"] for q in qa})
    rng.shuffle(patients)
    n_dev = max(1, int(0.2 * len(patients)))
    dev_pat = set(patients[:n_dev])
    train_pat = set(patients[n_dev:])

    def mine_negatives(qrow, maxk=bm25_topk):
        """Return candidate negative doc indices for this QA."""
        sid_gold = qrow["gold_section_id"]
        gold_row = sid_to_row.get(sid_gold, None)
        if gold_row is None:
            return []

        # Candidate pool: same patient (+ optional same report), excluding gold
        mask = (doc_pat == qrow["patient_id"])
        if restrict_same_report and qrow["gold_report_id"] is not None:
            mask &= (doc_rep == qrow["gold_report_id"])
        cand_idx = np.nonzero(mask)[0]
        cand_idx = cand_idx[cand_idx != gold_row]
        if len(cand_idx) == 0:
            return []

        scores = bm25.score(qrow["question"], cand_idx)
        order = np.argsort(-scores)
        top = cand_idx[order[:maxk]]
        return list(top)

    # Build pairs
    train_pairs: List[Pair] = []
    dev_pairs: List[Pair] = []

    for qrow in qa:
        sid_gold = qrow["gold_section_id"]
        gold_text = sec_text.get(sid_gold, None)
        if not gold_text:
            continue

        pos = Pair(qrow["question"], gold_text, 1.0)
        neg_rows = mine_negatives(qrow)
        if not neg_rows:
            continue

        # sample N negatives per positive (with fallback to random within pool)
        draw = neg_rows.copy()
        rng.shuffle(draw)
        draw = draw[:negatives_per_pos] if len(draw) >= negatives_per_pos else \
               (draw + rng.sample(neg_rows, k=min(negatives_per_pos - len(draw), len(neg_rows))))

        negs = [Pair(qrow["question"], docs[i], 0.0) for i in draw]

        if qrow["patient_id"] in dev_pat:
            dev_pairs.append(pos)
            dev_pairs.extend(negs)
        else:
            train_pairs.append(pos)
            train_pairs.extend(negs)

    return train_pairs, dev_pairs

# -----------------------------
# Dataset & collator
# -----------------------------

class PairDataset(Dataset):
    def __init__(self, pairs: List[Pair], tokenizer: AutoTokenizer, max_len: int = 384):
        self.pairs = pairs
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        enc = self.tok(
            p.q,
            p.d,
            truncation=True,
            padding=False,
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(p.label, dtype=torch.float)
        return item

def collate_fn(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if k == "labels":
            out[k] = torch.stack([b[k] for b in batch]).float()
        else:
            out[k] = torch.nn.utils.rnn.pad_sequence(
                [b[k] for b in batch],
                batch_first=True,
                padding_value=0,
            )
    return out

# -----------------------------
# Trainer with BCE loss
# -----------------------------

class BCETrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int = None,   
        **kwargs,                        
    ):
        labels = inputs.pop("labels").to(model.device)
        # ensure shape [B, 1] to match model logits
        if labels.dim() == 1:
            labels = labels.view(-1, 1)

        outputs = model(**inputs)
        logits = outputs.logits
        if logits.dim() == 1:
            logits = logits.view(-1, 1)

        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss
# -----------------------------
# Evaluation (rerank BM25 top-K)
# -----------------------------

def recall_at_k(ranks: List[int], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r < k) / max(1, len(ranks))

def mrr(ranks: List[int]) -> float:
    return sum(1.0/(r+1) for r in ranks if r is not None) / max(1, len(ranks))

def ndcg_at_k(rel_lists: List[List[int]], k: int) -> float:
    total = 0.0
    for rel in rel_lists:
        gains = [rel[i] / math.log2(i+2) for i in range(min(k, len(rel)))]
        dcg = sum(gains)
        ideal = sorted(rel, reverse=True)
        idcg = sum(ideal[i] / math.log2(i+2) for i in range(min(k, len(ideal))))
        total += (dcg / idcg) if idcg > 0 else 0.0
    return total / max(1, len(rel_lists))

@torch.no_grad()
def evaluate_reranker(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    qa_dev: List[dict],
    sec_text: Dict[int, str],
    sec_report: Dict[int, str],
    sec_patient: Dict[int, str],
    bm25_topk: int = 100,
    restrict_same_report: bool = False,
    max_len: int = 384,
    batch_size: int = 64,
):
    # Build BM25 index
    all_section_ids = sorted(sec_text.keys())
    sid_to_row = {sid: i for i, sid in enumerate(all_section_ids)}
    docs = [sec_text[sid] for sid in all_section_ids]
    doc_pat = np.array([sec_patient[sid] for sid in all_section_ids], dtype=object)
    doc_rep = np.array([sec_report[sid] for sid in all_section_ids], dtype=object)

    bm25 = BM25Index()
    bm25.add_docs(docs)

    ranks = []
    rel_lists = []
    Kmax = 20  # fixed for metrics below

    device = model.device
    model.eval()

    for q in qa_dev:
        gold_sid = q["gold_section_id"]
        gold_row = sid_to_row.get(gold_sid, None)
        if gold_row is None:
            ranks.append(None)
            rel_lists.append([0]*Kmax)
            continue

        mask = (doc_pat == q["patient_id"])
        if restrict_same_report and q["gold_report_id"] is not None:
            mask &= (doc_rep == q["gold_report_id"])
        cand_idx = np.nonzero(mask)[0]
        # Remove gold from pool (still want to be able to rank it)
        # We'll keep it; if you prefer to exclude, comment the following line.
        # (Keeping it is standard for reranking eval.)
        if len(cand_idx) == 0:
            ranks.append(None)
            rel_lists.append([0]*Kmax)
            continue

        # BM25 top-K -> rerank by cross-encoder
        bm_scores = bm25.score(q["question"], cand_idx)
        order = np.argsort(-bm_scores)
        cand_idx = cand_idx[order[:bm25_topk]]

        # Build batches of (q, doc)
        texts1, texts2 = [], []
        for di in cand_idx:
            sid = all_section_ids[di]
            texts1.append(q["question"])
            texts2.append(sec_text[sid])

        # Tokenize in chunks
        scores = []
        for i in range(0, len(texts1), batch_size):
            enc = tokenizer(
                texts1[i:i+batch_size],
                texts2[i:i+batch_size],
                truncation=True,
                padding=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits.view(-1)
            scores.append(logits.detach().cpu().numpy())
        if scores:
            scores = np.concatenate(scores, axis=0)
        else:
            scores = np.array([])

        # Final order by cross-encoder score
        rerank_order = np.argsort(-scores)
        top_rows = [all_section_ids[cand_idx[j]] for j in rerank_order[:Kmax]]
        # Convert to dataset row indices for metric calc
        # map back to global row index
        top_rows_global = [sid_to_row[sid] for sid in top_rows]

        gold_set = {gold_row}
        rr = None
        rel_list = []
        for rank, r in enumerate(top_rows_global):
            rel = 1 if r in gold_set else 0
            rel_list.append(rel)
            if rr is None and rel:
                rr = rank
        ranks.append(rr)
        # pad rel list for Kmax if needed
        if len(rel_list) < Kmax:
            rel_list = rel_list + [0]*(Kmax - len(rel_list))
        rel_lists.append(rel_list[:Kmax])

    results = {}
    for k in [1, 5, 10, 20]:
        results[f"recall@{k}"] = recall_at_k(ranks, k)
        results[f"ndcg@{k}"] = ndcg_at_k(rel_lists, k)
    results["mrr"] = mrr(ranks)
    return results

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--base-model", default="models/bge-reranker-v2-m3",
                    help="Cross-encoder backbone (multilingual).")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--per-device-train-batch-size", type=int, default=16)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)

    # negatives & mining
    ap.add_argument("--negatives-per-pos", type=int, default=4)
    ap.add_argument("--bm25-train-topk", type=int, default=50)
    ap.add_argument("--bm25-eval-topk", type=int, default=100)
    ap.add_argument("--restrict-same-report", type=int, default=0)

    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    conn = connect(args.db)
    try:
        # Load data
        qa = load_qa(conn)
        sec_text, sec_report, sec_section_name, sec_patient = load_corpus(conn)
        print(f"Loaded {len(qa)} QA items; {len(sec_text)} snippets.")

        # Build train/dev pairs (patient-held-out)
        train_pairs, dev_pairs = build_pairs(
            qa, sec_text, sec_report, sec_patient,
            negatives_per_pos=args.negatives_per_pos,
            bm25_topk=args.bm25_train_topk,
            restrict_same_report=bool(args.restrict_same_report),
            seed=args.seed
        )
        # Derive dev QA subset (those whose patient is in dev)
        dev_patients = {sec_patient[q["gold_section_id"]] for q in qa}
        # Actually recompute dev QA properly:
        pats = sorted({q["patient_id"] for q in qa})
        n_dev = max(1, int(0.2 * len(pats)))
        dev_pat = set(pats[:n_dev])
        qa_dev = [q for q in qa if q["patient_id"] in dev_pat]

        print(f"Train pairs: {len(train_pairs)}  Dev pairs: {len(dev_pairs)}  Dev QA: {len(qa_dev)}")

        # Model & tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model, num_labels=1
        )

        # Datasets
        train_ds = PairDataset(train_pairs, tokenizer, max_len=args.max_len)
        dev_ds = PairDataset(dev_pairs, tokenizer, max_len=args.max_len)

        # Training
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            logging_steps=50,
            save_strategy="epoch",
            eval_strategy="no",  # we do our own rerank eval after training epochs if desired
            fp16=torch.cuda.is_available(),
            bf16=False,
            dataloader_pin_memory=True,
            report_to=[],
            seed=args.seed,
        )

        trainer = BCETrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=None,
            tokenizer=tokenizer,
            data_collator=collate_fn,
        )

        trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Evaluation (reranking BM25 top-K on dev QA)
        model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
        model.to("cuda" if torch.cuda.is_available() else "cpu")

        eval_res = evaluate_reranker(
            model, tokenizer, qa_dev,
            sec_text, sec_report, sec_patient,
            bm25_topk=args.bm25_eval_topk,
            restrict_same_report=bool(args.restrict_same_report),
            max_len=args.max_len,
            batch_size=64,
        )
        print("\n=== Cross-Encoder Rerank Evaluation (patient-held-out dev) ===")
        for k in [1,5,10,20]:
            print(f"Recall@{k}: {eval_res[f'recall@{k}']:.3f}   nDCG@{k}: {eval_res[f'ndcg@{k}']:.3f}")
        print(f"MRR: {eval_res['mrr']:.3f}")

        # Save metrics
        with open(os.path.join(args.output_dir, "dev_metrics.json"), "w", encoding="utf-8") as f:
            import json
            json.dump(eval_res, f, indent=2)

    finally:
        conn.close()

if __name__ == "__main__":
    main()
