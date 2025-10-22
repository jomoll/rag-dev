#!/usr/bin/env python3
"""
Count number of files per patient with statistics
"""

import sqlite3
import statistics
from pathlib import Path
import sys

def count_files_per_patient(db_path: str):
    """Count files per patient and show statistics"""
    
    conn = sqlite3.connect(db_path)
    
    # Get file count per patient
    cursor = conn.execute("""
        SELECT patient_id, COUNT(*) as file_count 
        FROM reports 
        GROUP BY patient_id
        ORDER BY file_count DESC
    """)
    
    patient_counts = []
    print("Files per patient:")
    print("Patient ID | File Count")
    print("-" * 25)
    
    for patient_id, file_count in cursor:
        patient_counts.append(file_count)
        print(f"{patient_id:10} | {file_count:10}")
    
    conn.close()
    
    if patient_counts:
        print(f"\n=== STATISTICS ===")
        print(f"Total patients: {len(patient_counts)}")
        print(f"Total files: {sum(patient_counts)}")
        print(f"Min files per patient: {min(patient_counts)}")
        print(f"Max files per patient: {max(patient_counts)}")
        print(f"Mean files per patient: {statistics.mean(patient_counts):.2f}")
        print(f"Median files per patient: {statistics.median(patient_counts):.2f}")
        if len(patient_counts) > 1:
            print(f"Standard deviation: {statistics.stdev(patient_counts):.2f}")
    else:
        print("No data found!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python count_files.py <db_path>")
        print("Example: python count_files.py reports.db")
        sys.exit(1)
    
    db_path = sys.argv[1]
    
    if not Path(db_path).exists():
        print(f"Database file not found: {db_path}")
        sys.exit(1)
    
    count_files_per_patient(db_path)