#!/usr/bin/env python3
"""
Analyze existing reports to find all section names and their content lengths
"""

import sqlite3
import re
from collections import Counter
from pathlib import Path
import statistics

def analyze_report_sections(db_path: str):
    """Analyze all reports to find unique section names and content lengths"""
    
    conn = sqlite3.connect(db_path)
    
    # Get all report content
    cursor = conn.execute("SELECT report_id, content FROM reports")
    
    section_counter = Counter()
    section_lengths = []
    section_length_by_name = {}
    
    # For merged analysis
    merged_counter = Counter()
    merged_lengths = []
    merged_length_by_name = {}
    
    total_reports = 0
    reports_with_sections = 0
    
    print("Analyzing report sections and their lengths...")
    
    for report_id, content in cursor:
        total_reports += 1
        
        if not content:
            continue
            
        # Split content by section headers and process each section
        sections = extract_sections(content)
        
        if sections:
            reports_with_sections += 1
            
        # Process original sections
        for section_name, section_content in sections:
            section_counter[section_name] += 1
            content_length = len(section_content.strip().split())
            section_lengths.append(content_length)
            
            if section_name not in section_length_by_name:
                section_length_by_name[section_name] = []
            section_length_by_name[section_name].append(content_length)
        
        # Process merged sections (merge sections < 50 words with previous)
        merged_sections = merge_small_sections(sections, min_words=50)
        for section_name, section_content in merged_sections:
            merged_counter[section_name] += 1
            content_length = len(section_content.strip().split())
            merged_lengths.append(content_length)
            
            if section_name not in merged_length_by_name:
                merged_length_by_name[section_name] = []
            merged_length_by_name[section_name].append(content_length)
    
    conn.close()
    
    # Print original analysis
    print(f"\n=== ORIGINAL SECTION ANALYSIS ===")
    print(f"Total reports: {total_reports}")
    print(f"Reports with sections: {reports_with_sections}")
    print(f"Unique section names: {len(section_counter)}")
    print(f"Total sections found: {len(section_lengths)}")
    
    if section_lengths:
        print(f"\n=== ORIGINAL SECTION LENGTH STATISTICS (in words) ===")
        print(f"Mean section length: {statistics.mean(section_lengths):.1f} words")
        print(f"Median section length: {statistics.median(section_lengths):.1f} words")
        print(f"Max section length: {max(section_lengths):,} words")
        print(f"Min section length: {min(section_lengths):,} words")
        print(f"Standard deviation: {statistics.stdev(section_lengths):.1f} words")
        
        small_sections = sum(1 for length in section_lengths if length < 50)
        print(f"Sections < 50 words: {small_sections} ({(small_sections/len(section_lengths)*100):.1f}%)")
    
    # Print merged analysis
    print(f"\n=== MERGED SECTION ANALYSIS (< 50 words merged with previous) ===")
    print(f"Total sections after merge: {len(merged_lengths)}")
    print(f"Sections removed by merging: {len(section_lengths) - len(merged_lengths)}")
    
    if merged_lengths:
        print(f"\n=== MERGED SECTION LENGTH STATISTICS (in words) ===")
        print(f"Mean section length: {statistics.mean(merged_lengths):.1f} words")
        print(f"Median section length: {statistics.median(merged_lengths):.1f} words")
        print(f"Max section length: {max(merged_lengths):,} words")
        print(f"Min section length: {min(merged_lengths):,} words")
        print(f"Standard deviation: {statistics.stdev(merged_lengths):.1f} words")
        
        small_sections = sum(1 for length in merged_lengths if length < 50)
        print(f"Sections < 50 words: {small_sections} ({(small_sections/len(merged_lengths)*100):.1f}%)")
    
    # Compare the two approaches
    print(f"\n=== COMPARISON: ORIGINAL vs MERGED ===")
    print(f"{'Metric':<25} {'Original':<12} {'Merged':<12} {'Change'}")
    print(f"{'-'*60}")
    
    if section_lengths and merged_lengths:
        orig_mean = statistics.mean(section_lengths)
        merged_mean = statistics.mean(merged_lengths)
        print(f"{'Total sections:':<25} {len(section_lengths):<12} {len(merged_lengths):<12} {len(merged_lengths)-len(section_lengths):+d}")
        print(f"{'Mean length:':<25} {orig_mean:<12.1f} {merged_mean:<12.1f} {merged_mean-orig_mean:+.1f}")
        print(f"{'Max length:':<25} {max(section_lengths):<12,} {max(merged_lengths):<12,} {max(merged_lengths)-max(section_lengths):+,}")
        print(f"{'Min length:':<25} {min(section_lengths):<12,} {min(merged_lengths):<12,} {min(merged_lengths)-min(section_lengths):+,}")
    
    print(f"\n=== TOP SECTION NAMES AFTER MERGING (sorted by frequency) ===")
    for section_name, count in merged_counter.most_common(20):
        lengths = merged_length_by_name[section_name]
        mean_len = statistics.mean(lengths)
        max_len = max(lengths)
        print(f"{count:6d} | {mean_len:7.1f} avg | {max_len:7,} max | {section_name}")
    
    return section_counter, section_lengths, merged_counter, merged_lengths

def merge_small_sections(sections, min_words=50):
    """Merge sections with < min_words into the previous section, or next if it's the first"""
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
                # Not the first section - merge with previous (existing logic)
                prev_name, prev_content = merged[-1]
                combined_content = f"{prev_content}\n\n## {section_name}:\n{section_content}"
                merged[-1] = (prev_name, combined_content)
                i += 1
            else:
                # First section or start of a chain - keep merging with next sections until we hit min_words
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

def extract_sections(content: str):
    """Extract sections from report content"""
    sections = []
    
    # Split by section headers (##)
    parts = re.split(r'^(##\s*.+?)$', content, flags=re.MULTILINE)
    
    current_section_name = None
    
    for i, part in enumerate(parts):
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

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python test4.py <db_path>")
        print("Example: python test4.py reports.db")
        sys.exit(1)
    
    db_path = sys.argv[1]
    
    if not Path(db_path).exists():
        print(f"Database file not found: {db_path}")
        sys.exit(1)
    
    analyze_report_sections(db_path)