#!/usr/bin/env python3
"""
Script to rename moved files by adding category keywords in parentheses.
Reads the audit CSV and renames files in patient directories.

Usage: python add_category_to_filenames.py <audit_csv_path>
"""

import csv
import sys
from pathlib import Path
from datetime import datetime

# Mapping from folder_prefix to category keyword
CATEGORY_MAPPING = {
    "Arzbriefe Ambulanz bis 2020": "Arztbrief",
    "Arztbriefe Ambulanz bis 2020": "Arztbrief", 
    "Arztbriefe": "Arztbrief",
    "Cytology": "Cytologie", 
    "Flow-Cytology": "Flow Cytometry",
    "Flow cytology": "Flow Cytometry"
}

def add_category_to_filename(filename: str, category: str) -> str:
    """Add category in parentheses before file extension"""
    path = Path(filename)
    stem = path.stem
    suffix = path.suffix
    return f"{stem} ({category}){suffix}"

def rename_with_category(audit_csv_path: Path):
    """Rename files by adding category keywords in parentheses"""
    
    if not audit_csv_path.exists():
        print(f"Error: Audit file not found: {audit_csv_path}")
        return
    
    renamed_count = 0
    error_count = 0
    
    print(f"Reading audit log: {audit_csv_path}")
    
    with audit_csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            action = row.get('action', '')
            new_path = row.get('new_path', '')
            folder_prefix = row.get('folder_prefix', '')
            
            # Only process moved files (not ignored extras)
            if action != 'move+rename' or not new_path or not folder_prefix:
                continue
            
            current_file = Path(new_path)
            if not current_file.exists():
                print(f"Warning: File not found: {current_file}")
                error_count += 1
                continue
            
            # Get category from folder_prefix
            category = CATEGORY_MAPPING.get(folder_prefix)
            if not category:
                print(f"Warning: Unknown folder prefix '{folder_prefix}' for file: {current_file}")
                # Default fallback
                category = "Unknown"
            
            try:
                # Skip if already has parentheses (already processed)
                original_filename = current_file.name
                if '(' in original_filename and ')' in original_filename:
                    print(f"Skipping (already has category): {original_filename}")
                    continue
                
                # Create new filename with category
                new_filename = add_category_to_filename(original_filename, category)
                new_file_path = current_file.parent / new_filename
                
                # Avoid overwriting existing files
                counter = 1
                while new_file_path.exists():
                    stem = Path(original_filename).stem
                    suffix = Path(original_filename).suffix
                    numbered_filename = f"{stem} ({category}) {counter}{suffix}"
                    new_file_path = current_file.parent / numbered_filename
                    counter += 1
                
                # Rename the file
                current_file.rename(new_file_path)
                print(f"Renamed: {original_filename} -> {new_file_path.name}")
                renamed_count += 1
                
            except Exception as e:
                print(f"Error processing {current_file}: {e}")
                error_count += 1
    
    print(f"\nComplete! Renamed {renamed_count} files, {error_count} errors")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python add_category_to_filenames.py <audit_csv_path>")
        sys.exit(1)
    
    audit_path = Path(sys.argv[1])
    rename_with_category(audit_path)