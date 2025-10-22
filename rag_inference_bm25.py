#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG Inference Pipeline for Medical Reports with BM25 Retrieval

Retrieves k most relevant chunks for a patient based on BM25 similarity,
then uses them as context for answering questions via Llama 3.3 70B.

Usage:
    python rag_inference_bm25.py \
        --db myeloma_reports_de.sqlite \
        --patient-id "0001005585" \
        --question "Was sind die aktuellen Befunde?" \
        --top-k 5 \
        --model llama-3.3-70b
"""

import argparse
import asyncio
import contextlib
import json
import math
import re
import sqlite3
import time
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Any
import logging

import httpx
import numpy as np

# ---------------------------
# Configuration
# ---------------------------

SYSTEM_PROMPT = """Du bist ein medizinischer Assistent, der Fragen zu Patientenberichten beantwortet.
Du erhältst relevante Textabschnitte aus den Berichten eines Patienten als Kontext.
Antworte präzise und basiere deine Antworten nur auf den gegebenen Kontextinformationen.
Wenn die Antwort nicht aus dem Kontext hervorgeht, sage das deutlich.
Verwende eine professionelle, medizinische Sprache."""

USER_PROMPT_TEMPLATE = """Patient ID: {patient_id}

Relevante Befunde (sortiert nach Relevanz):

{context_chunks}

Frage: {question}

Bitte beantworte die Frage basierend auf den oben gegebenen Befunden. Erwähne die Quellen (Bericht und Abschnitt) für deine Antworten."""

DEFAULT_MODEL_NAME = "llama-3.3-70b-instruct"

# ---------------------------
# Logging
# ---------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------
# BM25 Implementation (from compute_retrieval_metrics.py)
# ---------------------------

def _normalize_text(s: str) -> str:
    return (s or "").lower()

def _char_ngrams(s: str, n: int = 3) -> List[str]:
    s = re.sub(r"[^0-9a-zA-ZäöüÄÖÜß]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(" ", "_")  # keep some boundary info
    if len(s) < n:
        return [s] if s else []
    return [s[i:i+n] for i in range(len(s)-n+1)]

class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75, ngram: int = 3):
        self.k1 = k1
        self.b = b
        self.ngram = ngram
        self.df = Counter()
        self.doc_len = []
        self.avgdl = 0.0
        self.N = 0
        self.postings: Dict[str, Dict[int, int]] = defaultdict(dict)

    def add_docs(self, docs: List[str]):
        self.N = len(docs)
        for doc_id, text in enumerate(docs):
            toks = _char_ngrams(_normalize_text(text), self.ngram)
            tf = Counter(toks)
            self.doc_len.append(sum(tf.values()))
            for t, c in tf.items():
                self.df[t] += 1
                self.postings[t][doc_id] = c
        self.avgdl = (sum(self.doc_len) / max(1, self.N)) if self.N else 0.0

    def score(self, query: str, cand_mask: np.ndarray) -> np.ndarray:
        toks = _char_ngrams(_normalize_text(query), self.ngram)
        unique_q = set(toks)
        # precompute IDF
        idf = {}
        for t in unique_q:
            df_t = self.df.get(t, 0)
            # standard BM25 IDF with +0.5 smoothing
            idf[t] = math.log((self.N - df_t + 0.5) / (df_t + 0.5) + 1.0) if self.N else 0.0

        idxs = np.nonzero(cand_mask)[0]
        scores = np.zeros(len(idxs), dtype=np.float32)
        for j, doc_id in enumerate(idxs):
            dl = self.doc_len[doc_id] if doc_id < len(self.doc_len) else 0
            denom_norm = self.k1 * (1 - self.b + self.b * (dl / self.avgdl)) if self.avgdl > 0 else self.k1
            s = 0.0
            for t in unique_q:
                tf = self.postings.get(t, {}).get(doc_id, 0)
                if tf == 0:
                    continue
                num = tf * (self.k1 + 1.0)
                s += idf[t] * (num / (tf + denom_norm))
            scores[j] = s
        return scores

# ---------------------------
# Database Connection
# ---------------------------

def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

# ---------------------------
# BM25-based Retrieval Functions
# ---------------------------

class PatientBM25Retriever:
    def __init__(self, conn: sqlite3.Connection, k1: float = 1.2, b: float = 0.75, ngram: int = 3):
        self.conn = conn
        self.k1 = k1
        self.b = b
        self.ngram = ngram
        self.bm25_index = None
        self.chunks_data = []
        self.patient_indices = {}  # patient_id -> list of chunk indices
        
    def build_index_for_patient(self, patient_id: str):
        """Build BM25 index for a specific patient's chunks"""
        logger.info(f"Building BM25 index for patient {patient_id}")
        
        # Load all chunks for this patient
        rows = self.conn.execute("""
            SELECT 
                rs.section_id,
                rs.report_id,
                rs.name as section_name,
                rs.text,
                rs.chunk_index,
                r.created_at,
                r.source_path
            FROM report_sections rs
            JOIN reports r ON rs.report_id = r.report_id
            WHERE r.patient_id = ?
            AND rs.text IS NOT NULL
            AND LENGTH(rs.text) > 50
            ORDER BY r.created_at, rs.section_id, rs.chunk_index
        """, (patient_id,)).fetchall()
        
        if not rows:
            logger.warning(f"No chunks found for patient {patient_id}")
            return False
        
        # Store chunk metadata
        self.chunks_data = []
        docs = []
        
        for row in rows:
            chunk_data = {
                'section_id': row['section_id'],
                'report_id': row['report_id'],
                'section_name': row['section_name'],
                'text': row['text'],
                'chunk_index': row['chunk_index'],
                'created_at': row['created_at'],
                'source_path': row['source_path']
            }
            self.chunks_data.append(chunk_data)
            docs.append(row['text'])
        
        # Build BM25 index
        self.bm25_index = BM25Index(k1=self.k1, b=self.b, ngram=self.ngram)
        self.bm25_index.add_docs(docs)
        
        logger.info(f"Built BM25 index with {len(docs)} chunks for patient {patient_id}")
        return True
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant chunks using BM25"""
        if not self.bm25_index or not self.chunks_data:
            logger.error("BM25 index not built. Call build_index_for_patient first.")
            return []
        
        # Create mask for all documents (no filtering since we already filtered by patient)
        cand_mask = np.ones(len(self.chunks_data), dtype=bool)
        
        # Get BM25 scores
        scores = self.bm25_index.score(query, cand_mask)
        
        # Get top-k indices
        order = np.argsort(-scores)
        top_indices = order[:top_k]
        
        # Prepare results
        results = []
        for i, idx in enumerate(top_indices):
            chunk = self.chunks_data[idx].copy()
            chunk['similarity'] = float(scores[idx])
            chunk['rank'] = i + 1
            results.append(chunk)
        
        return results

def retrieve_chunks_by_bm25(
    conn: sqlite3.Connection,
    patient_id: str,
    query: str,
    top_k: int = 5,
    k1: float = 1.2,
    b: float = 0.75,
    ngram: int = 3
) -> List[Dict[str, Any]]:
    """Retrieve chunks using BM25 similarity"""
    
    retriever = PatientBM25Retriever(conn, k1=k1, b=b, ngram=ngram)
    
    # Build index for this patient
    if not retriever.build_index_for_patient(patient_id):
        return []
    
    # Search
    results = retriever.search(query, top_k)
    
    logger.info(f"BM25 retrieved {len(results)} chunks for query: {query[:100]}...")
    for i, result in enumerate(results[:3]):  # Log top 3
        logger.info(f"  {i+1}. Score: {result['similarity']:.3f} | {result['section_name']} | Text: {result['text'][:100]}...")
    
    return results

def retrieve_chunks_by_fts(
    conn: sqlite3.Connection, 
    patient_id: str, 
    query: str, 
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Fallback: retrieve chunks using full-text search"""
    
    sql_query = """
        SELECT 
            rs.section_id,
            rs.report_id,
            rs.name as section_name,
            rs.text,
            rs.chunk_index,
            r.created_at,
            r.source_path,
            fts.rank
        FROM report_fts fts
        JOIN report_sections rs ON rs.section_id = fts.rowid
        JOIN reports r ON rs.report_id = r.report_id
        WHERE fts MATCH ? 
        AND r.patient_id = ?
        AND LENGTH(rs.text) > 50
        ORDER BY fts.rank
        LIMIT ?
    """
    
    cursor = conn.execute(sql_query, (query, patient_id, top_k))
    
    results = []
    for row in cursor.fetchall():
        results.append({
            'section_id': row['section_id'],
            'report_id': row['report_id'],
            'section_name': row['section_name'],
            'text': row['text'],
            'chunk_index': row['chunk_index'],
            'created_at': row['created_at'],
            'source_path': row['source_path'],
            'similarity': 1.0 / (1.0 + row['rank'])  # Convert rank to similarity-like score
        })
    
    return results

def format_context_chunks(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks for LLM context"""
    if not chunks:
        return "Keine relevanten Befunde gefunden."
    
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        # Extract date from created_at or source_path
        date_info = chunk.get('created_at', 'Unbekanntes Datum')
        if chunk.get('source_path'):
            # Try to extract date from filename if available
            date_match = re.search(r'(\d{1,2}\.\d{1,2}\.20\d{2})', chunk['source_path'])
            if date_match:
                date_info = date_match.group(1)
        
        # For BM25, show BM25 score instead of similarity
        score_info = f"BM25: {chunk['similarity']:.3f}" if chunk['similarity'] > 0 else "Score: N/A"
        
        context_parts.append(
            f"[{i}] Bericht: {chunk['report_id']} | "
            f"Abschnitt: {chunk['section_name']} | "
            f"Datum: {date_info} | "
            f"{score_info}\n"
            f"{chunk['text'].strip()}\n"
        )
    
    return "\n".join(context_parts)

# ---------------------------
# LLM Client (unchanged from original)
# ---------------------------

def make_async_client(evaluation_model: str) -> tuple[httpx.AsyncClient, str]:
    """Create async HTTP client for LLM endpoint"""
    if evaluation_model == "llama-3.3-70b":
        BASE_URL = "http://localhost:9999/v1"
        TGI_TOKEN = "dacbebe8c973154018a3d0f5"
        HEADERS = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TGI_TOKEN}",
        }
        TIMEOUT = httpx.Timeout(connect=15.0, read=180.0, write=15.0, pool=15.0)
        client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers=HEADERS,
            timeout=TIMEOUT,
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=2)
        )
        return client, DEFAULT_MODEL_NAME
    
    raise ValueError(f"Unknown evaluation_model={evaluation_model}")

async def call_llm(
    client: httpx.AsyncClient, 
    patient_id: str,
    question: str, 
    context_chunks: str,
    model_name: str,
    max_tokens: int = 1000,
    temperature: float = 0.1
) -> str:
    """Call LLM with context and question"""
    
    user_prompt = USER_PROMPT_TEMPLATE.format(
        patient_id=patient_id,
        context_chunks=context_chunks,
        question=question
    )
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": 1,
        "stop": None,
        "stream": False,
    }
    
    logger.debug(f"Making LLM request, prompt length: {len(user_prompt)}")
    
    resp = await client.post("/chat/completions", json=payload)
    resp.raise_for_status()
    
    data = resp.json()
    content = data["choices"][0]["message"]["content"].strip()
    
    return content

# ---------------------------
# Main Functions
# ---------------------------

async def run_inference(
    db_path: str,
    patient_id: str,
    question: str,
    top_k: int = 5,
    model_name: str = "llama-3.3-70b",
    use_fts_fallback: bool = True,
    bm25_k1: float = 1.2,
    bm25_b: float = 0.75,
    bm25_ngram: int = 3
) -> Dict[str, Any]:
    """Run RAG inference pipeline with BM25 retrieval"""
    
    conn = connect(db_path)
    
    # Check if patient exists
    patient_check = conn.execute(
        "SELECT COUNT(*) as count FROM reports WHERE patient_id = ?", 
        (patient_id,)
    ).fetchone()
    
    if patient_check['count'] == 0:
        conn.close()
        return {
            "error": f"Patient {patient_id} not found in database",
            "patient_id": patient_id,
            "question": question
        }
    
    logger.info(f"Found {patient_check['count']} reports for patient {patient_id}")
    
    # Retrieve relevant chunks using BM25
    logger.info(f"Retrieving top {top_k} chunks using BM25 for query: {question[:100]}...")
    
    try:
        chunks = retrieve_chunks_by_bm25(
            conn, patient_id, question, top_k,
            k1=bm25_k1, b=bm25_b, ngram=bm25_ngram
        )
        retrieval_method = "bm25_char3gram"
        
        if not chunks and use_fts_fallback:
            logger.info("No chunks found with BM25, falling back to FTS")
            chunks = retrieve_chunks_by_fts(conn, patient_id, question, top_k)
            retrieval_method = "full_text_search"
            
    except Exception as e:
        logger.error(f"Error in BM25 retrieval: {e}")
        if use_fts_fallback:
            logger.info("Falling back to FTS due to error")
            chunks = retrieve_chunks_by_fts(conn, patient_id, question, top_k)
            retrieval_method = "full_text_search_fallback"
        else:
            conn.close()
            raise
    
    if not chunks:
        conn.close()
        return {
            "error": "No relevant chunks found for the query",
            "patient_id": patient_id,
            "question": question,
            "retrieval_method": retrieval_method
        }
    
    logger.info(f"Retrieved {len(chunks)} chunks using {retrieval_method}")
    
    # Format context
    context_text = format_context_chunks(chunks)
    
    # Call LLM
    client, llm_model_name = make_async_client(model_name)
    
    try:
        logger.info("Generating response with LLM...")
        start_time = time.time()
        
        response = await call_llm(
            client, patient_id, question, context_text, llm_model_name
        )
        
        inference_time = time.time() - start_time
        logger.info(f"LLM response generated in {inference_time:.2f}s")
        
        result = {
            "patient_id": patient_id,
            "question": question,
            "answer": response,
            "retrieval_method": retrieval_method,
            "chunks_retrieved": len(chunks),
            "inference_time_seconds": inference_time,
            "bm25_params": {
                "k1": bm25_k1,
                "b": bm25_b,
                "ngram": bm25_ngram
            },
            "context_chunks": [
                {
                    "report_id": chunk["report_id"],
                    "section_name": chunk["section_name"],
                    "bm25_score": chunk["similarity"],
                    "text_preview": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
                }
                for chunk in chunks
            ]
        }
        
        return result
        
    finally:
        await client.aclose()
        conn.close()

# ---------------------------
# CLI Interface
# ---------------------------

async def main_async(args):
    """Main async function"""
    result = await run_inference(
        db_path=args.db,
        patient_id=args.patient_id,
        question=args.question,
        top_k=args.top_k,
        model_name=args.model,
        use_fts_fallback=not args.no_fts_fallback,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        bm25_ngram=args.bm25_ngram
    )
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    # Print results
    print(f"\n🔍 RAG Inference Results (BM25)")
    print(f"{'='*50}")
    print(f"Patient ID: {result['patient_id']}")
    print(f"Question: {result['question']}")
    print(f"Retrieval: {result['retrieval_method']} ({result['chunks_retrieved']} chunks)")
    print(f"BM25 params: k1={result['bm25_params']['k1']}, b={result['bm25_params']['b']}, ngram={result['bm25_params']['ngram']}")
    print(f"Response time: {result['inference_time_seconds']:.2f}s")
    print(f"\n📋 Retrieved Context Sources:")
    for i, chunk in enumerate(result['context_chunks'], 1):
        print(f"  [{i}] {chunk['report_id']} | {chunk['section_name']} | BM25: {chunk['bm25_score']:.3f}")
    
    print(f"\n🤖 Answer:")
    print(f"{'-'*50}")
    print(result['answer'])
    print(f"{'-'*50}")
    
    if args.save_result:
        output_file = f"rag_bm25_result_{result['patient_id']}_{int(time.time())}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Full result saved to: {output_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="RAG Inference for Medical Reports using BM25")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--patient-id", required=True, help="Patient ID to query")
    parser.add_argument("--question", required=True, help="Question to ask about the patient")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--model", default="llama-3.3-70b", help="LLM model to use")
    parser.add_argument("--no-fts-fallback", action="store_true", 
                       help="Disable full-text search fallback")
    parser.add_argument("--save-result", action="store_true", 
                       help="Save full result to JSON file")
    
    # BM25 parameters
    parser.add_argument("--bm25-k1", type=float, default=1.2, help="BM25 k1 parameter")
    parser.add_argument("--bm25-b", type=float, default=0.75, help="BM25 b parameter")
    parser.add_argument("--bm25-ngram", type=int, default=3, help="BM25 n-gram size")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main_async(args))