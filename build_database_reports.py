#!/usr/bin/env python3
"""
Add report sections table to existing database with merged sections and preserved metadata
"""

import sqlite3
import re
import datetime
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

def init_sections_table(conn: sqlite3.Connection):
    """Drop and create the report_sections table and FTS index"""
    
    conn.executescript("""
    DROP TABLE IF EXISTS report_sections;
    DROP TABLE IF EXISTS sections_fts;

    CREATE TABLE IF NOT EXISTS report_sections(
        section_id TEXT PRIMARY KEY,
        report_id TEXT NOT NULL,
        patient_id TEXT NOT NULL,
        section_name TEXT NOT NULL,
        section_content TEXT NOT NULL,
        section_order INTEGER NOT NULL,
        word_count INTEGER,
        -- Inherited metadata from reports
        filename TEXT,
        report_type TEXT,
        report_date TEXT,
        created_at TEXT,
        source_path TEXT,
        FOREIGN KEY(report_id) REFERENCES reports(report_id),
        FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
    );
    
    CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts USING fts5(
        section_id UNINDEXED,
        report_id UNINDEXED, 
        patient_id UNINDEXED,
        section_name UNINDEXED,
        filename UNINDEXED,
        report_type UNINDEXED,
        section_content,
        tokenize='porter'
    );
    
    CREATE INDEX IF NOT EXISTS idx_sections_patient ON report_sections(patient_id);
    CREATE INDEX IF NOT EXISTS idx_sections_report ON report_sections(report_id);
    CREATE INDEX IF NOT EXISTS idx_sections_type ON report_sections(section_name);
    CREATE INDEX IF NOT EXISTS idx_sections_date ON report_sections(report_date);
    """)
    
    conn.commit()

def extract_sections(content: str) -> List[Tuple[str, str]]:
    """Extract sections from report content"""
    sections = []
    
    # Split by section headers (##)
    parts = re.split(r'^(##\s*.+?)$', content, flags=re.MULTILINE)
    
    current_section_name = None
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # Check if this part is a section header
        if part.startswith('##'):
            # Extract section name
            section_line = part[2:].strip()  # Remove ##
            
            if ':' in section_line:
                section_name = section_line.split(':', 1)[0].strip()
            else:
                section_name = section_line.split()[0] if section_line.split() else section_line
            
            current_section_name = section_name
            
        elif current_section_name:
            # This is section content
            sections.append((current_section_name, part))
            current_section_name = None  # Reset for next section
    
    return sections

def merge_small_sections(sections: List[Tuple[str, str]], min_words: int = 50) -> List[Tuple[str, str]]:
    """Merge sections with < min_words into adjacent sections"""
    if not sections:
        return sections
    
    merged = []
    i = 0
    
    while i < len(sections):
        section_name, section_content = sections[i]
        word_count = len(section_content.strip().split())
        
        # If section is too small
        if word_count < min_words:
            if merged:
                # Not the first section - merge with previous
                prev_name, prev_content = merged[-1]
                combined_content = f"{prev_content}\n\n## {section_name}:\n{section_content}"
                merged[-1] = (prev_name, combined_content)
                i += 1
            else:
                # First section - keep merging with next sections until we hit min_words
                combined_name = section_name
                combined_content = f"## {section_name}:\n{section_content}"
                current_word_count = word_count
                i += 1
                
                # Keep adding following sections until we reach min_words
                while current_word_count < min_words and i < len(sections):
                    next_name, next_content = sections[i]
                    next_word_count = len(next_content.strip().split())
                    
                    combined_content += f"\n\n## {next_name}:\n{next_content}"
                    current_word_count += next_word_count
                    i += 1
                
                merged.append((combined_name, combined_content))
        else:
            # Section is large enough - keep as separate section
            merged.append((section_name, section_content))
            i += 1
    
    return merged

def generate_section_id(report_id: str, section_name: str, section_order: int) -> str:
    """Generate unique section ID"""
    # Clean section name for ID
    clean_name = re.sub(r'[^A-Za-z0-9._-]', '_', section_name)[:30]
    return f"{report_id}_sec_{section_order:03d}_{clean_name}"

def split_section_with_overlap(section_name: str, section_content: str, max_words: int = 350, overlap: int = 50):
    """Split section content into chunks with overlap if longer than max_words"""
    words = section_content.strip().split()
    if len(words) <= max_words:
        return [(section_name, section_content)]
    chunks = []
    start = 0
    chunk_id = 1
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]
        chunk_text = ' '.join(chunk_words)
        chunk_name = f"{section_name} (part {chunk_id})"
        chunks.append((chunk_name, chunk_text))
        if end == len(words):
            break
        start = end - overlap
        chunk_id += 1
    return chunks

def process_report_sections(conn: sqlite3.Connection, db_path: str):
    """Process all reports and extract sections"""
    
    # Get all reports with metadata
    cursor = conn.execute("""
        SELECT r.report_id, r.patient_id, r.filename, r.report_type, 
               r.report_date, r.created_at, r.source_path, r.content
        FROM reports r
        ORDER BY r.patient_id, r.report_date
    """)
    
    reports = cursor.fetchall()
    total_sections = 0
    processed_reports = 0
    
    print(f"Processing {len(reports)} reports...")
    
    for report_data in tqdm(reports, desc="Extracting sections"):
        (report_id, patient_id, filename, report_type, 
         report_date, created_at, source_path, content) = report_data
        
        if not content or not content.strip():
            continue

        # Skip "labor" reports (JSON lab values)
        if report_type and report_type.lower() == "labor":
            continue
            
        # Extract and merge sections
        raw_sections = extract_sections(content)
        merged_sections = merge_small_sections(raw_sections, min_words=50)
        
        # If no sections found, treat the whole report as a single section
        if not merged_sections:
            merged_sections = [("Full report", content.strip())]
            
        # Insert sections
        for section_order, (section_name, section_content) in enumerate(merged_sections, 1):
            # Split if section is too long
            split_sections = split_section_with_overlap(section_name, section_content, max_words=350, overlap=50)
            for split_idx, (split_name, split_content) in enumerate(split_sections, 1):
                section_id = generate_section_id(report_id, split_name, section_order * 1000 + split_idx)
                word_count = len(split_content.strip().split())
                # Insert into report_sections table
                conn.execute("""
                    INSERT OR REPLACE INTO report_sections(
                        section_id, report_id, patient_id, section_name, section_content,
                        section_order, word_count, filename, report_type, report_date,
                        created_at, source_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    section_id, report_id, patient_id, split_name, split_content,
                    section_order, word_count, filename, report_type, report_date,
                    created_at, source_path
                ))
                # Insert into FTS index
                conn.execute("""
                    INSERT OR REPLACE INTO sections_fts(
                        section_id, report_id, patient_id, section_name, 
                        filename, report_type, section_content
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    section_id, report_id, patient_id, split_name,
                    filename, report_type, split_content
                ))
                total_sections += 1
        
        processed_reports += 1
        
        # Commit every 100 reports
        if processed_reports % 100 == 0:
            conn.commit()
    
    # Final commit
    conn.commit()
    
    return processed_reports, total_sections

def print_summary(conn: sqlite3.Connection):
    """Print summary statistics"""
    
    # Get counts
    section_count = conn.execute("SELECT COUNT(*) FROM report_sections").fetchone()[0]
    patient_count = conn.execute("SELECT COUNT(DISTINCT patient_id) FROM report_sections").fetchone()[0]
    report_count = conn.execute("SELECT COUNT(DISTINCT report_id) FROM report_sections").fetchone()[0]
    
    print(f"\n=== SECTIONS TABLE SUMMARY ===")
    print(f"Total sections: {section_count:,}")
    print(f"Reports with sections: {report_count:,}")
    print(f"Patients with sections: {patient_count:,}")
    
    # Section name distribution
    print(f"\n=== TOP SECTION TYPES ===")
    section_types = conn.execute("""
        SELECT section_name, COUNT(*) as count, 
               AVG(word_count) as avg_words, MAX(word_count) as max_words
        FROM report_sections 
        GROUP BY section_name 
        ORDER BY COUNT(*) DESC 
        LIMIT 15
    """).fetchall()
    
    for section_name, count, avg_words, max_words in section_types:
        print(f"{count:6,} | {avg_words:7.1f} avg | {max_words:7,} max | {section_name}")
    
    # Word count statistics
    word_counts = conn.execute("SELECT word_count FROM report_sections").fetchall()
    if word_counts:
        word_counts = [w[0] for w in word_counts]
        print(f"\n=== WORD COUNT STATISTICS ===")
        print(f"Mean: {sum(word_counts)/len(word_counts):.1f} words")
        print(f"Min: {min(word_counts):,} words")
        print(f"Max: {max(word_counts):,} words")
        
        under_50 = sum(1 for w in word_counts if w < 50)
        print(f"Sections < 50 words: {under_50} ({under_50/len(word_counts)*100:.1f}%)")

def build_sections_table(db_path: str):
    """Main function to build the sections table"""
    
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")
    
    print(f"Opening database: {db_path}")
    conn = sqlite3.connect(db_path, timeout=30.0)
    
    try:
        # Initialize sections table
        print("Creating sections table and indexes...")
        init_sections_table(conn)
        
        # Process reports and extract sections
        processed_reports, total_sections = process_report_sections(conn, db_path)
        
        print(f"\nProcessed {processed_reports:,} reports")
        print(f"Created {total_sections:,} sections")
        
        # Print summary
        print_summary(conn)
        
    finally:
        conn.close()
    
    print(f"\nSections table successfully added to: {db_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python add_sections_table.py <db_path>")
        print("Example: python add_sections_table.py reports.db")
        sys.exit(1)
    
    db_path = sys.argv[1]
    
    build_sections_table(db_path)