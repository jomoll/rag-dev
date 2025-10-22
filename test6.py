#!/usr/bin/env python3
"""
Clean up database by removing patients without names and all their associated data
"""

import sqlite3
from pathlib import Path

def get_existing_tables(conn):
    """Get list of existing tables in the database"""
    cursor = conn.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    """)
    return [row[0] for row in cursor.fetchall()]

def cleanup_database(db_path: str):
    """Remove patients without names and all their associated data"""
    
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")
    
    print(f"Opening database: {db_path}")
    conn = sqlite3.connect(db_path, timeout=30.0)
    
    try:
        # Check what tables exist
        existing_tables = get_existing_tables(conn)
        print(f"\nExisting tables: {', '.join(existing_tables)}")
        
        # First, let's see what we're dealing with
        cursor = conn.execute("""
            SELECT COUNT(*) as total_patients,
                   COUNT(CASE WHEN fullname IS NULL OR fullname = '' THEN 1 END) as patients_without_names,
                   COUNT(CASE WHEN fullname IS NOT NULL AND fullname != '' THEN 1 END) as patients_with_names
            FROM patients
        """)
        stats = cursor.fetchone()
        
        print(f"\n=== CURRENT DATABASE STATS ===")
        print(f"Total patients: {stats[0]:,}")
        print(f"Patients without names: {stats[1]:,}")
        print(f"Patients with names: {stats[2]:,}")
        
        # Get patients without names
        cursor = conn.execute("""
            SELECT patient_id FROM patients 
            WHERE fullname IS NULL OR fullname = '' OR fullname = ' '
        """)
        patients_to_remove = [row[0] for row in cursor.fetchall()]
        
        if not patients_to_remove:
            print("✅ No patients without names found. Database is clean.")
            return
        
        print(f"\n🗑️  Found {len(patients_to_remove)} patients without names to remove")
        
        # Show sample of patients to be removed
        print("\nSample patient IDs to be removed:")
        for i, patient_id in enumerate(patients_to_remove[:10]):
            print(f"  - {patient_id}")
        if len(patients_to_remove) > 10:
            print(f"  ... and {len(patients_to_remove) - 10} more")
        
        # Get counts of associated data before deletion
        placeholders = ','.join(['?' for _ in patients_to_remove])
        
        report_count = 0
        section_count = 0
        
        if 'reports' in existing_tables:
            report_count = conn.execute(f"""
                SELECT COUNT(*) FROM reports WHERE patient_id IN ({placeholders})
            """, patients_to_remove).fetchone()[0]
        
        if 'report_sections' in existing_tables:
            section_count = conn.execute(f"""
                SELECT COUNT(*) FROM report_sections WHERE patient_id IN ({placeholders})
            """, patients_to_remove).fetchone()[0]
        
        print(f"\n📊 Associated data to be removed:")
        print(f"Reports: {report_count:,}")
        print(f"Report sections: {section_count:,}")
        
        # Confirm deletion
        response = input(f"\n⚠️  This will permanently delete {len(patients_to_remove)} patients and all their data. Continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("❌ Deletion cancelled.")
            return
        
        print("\n🗑️  Starting cleanup...")
        
        # Delete in order: sections -> reports -> patients
        # This respects foreign key constraints
        
        if 'sections_fts' in existing_tables:
            print("1. Removing from sections_fts...")
            conn.execute(f"""
                DELETE FROM sections_fts WHERE section_id IN (
                    SELECT section_id FROM report_sections WHERE patient_id IN ({placeholders})
                )
            """, patients_to_remove)
        
        if 'report_sections' in existing_tables:
            print("2. Removing report sections...")
            conn.execute(f"""
                DELETE FROM report_sections WHERE patient_id IN ({placeholders})
            """, patients_to_remove)
        
        if 'reports_fts' in existing_tables:
            print("3. Removing from reports_fts...")
            conn.execute(f"""
                DELETE FROM reports_fts WHERE report_id IN (
                    SELECT report_id FROM reports WHERE patient_id IN ({placeholders})
                )
            """, patients_to_remove)
        
        if 'reports' in existing_tables:
            print("4. Removing reports...")
            conn.execute(f"""
                DELETE FROM reports WHERE patient_id IN ({placeholders})
            """, patients_to_remove)
        
        print("5. Removing patients...")
        conn.execute(f"""
            DELETE FROM patients WHERE patient_id IN ({placeholders})
        """, patients_to_remove)
        
        # Commit changes
        conn.commit()
        
        # Vacuum to reclaim space
        print("6. Vacuuming database to reclaim space...")
        conn.execute("VACUUM")
        
        # Get final stats
        cursor = conn.execute("""
            SELECT COUNT(*) as total_patients FROM patients
        """)
        remaining_patients = cursor.fetchone()[0]
        
        remaining_reports = 0
        remaining_sections = 0
        
        if 'reports' in existing_tables:
            cursor = conn.execute("""
                SELECT COUNT(*) as total_reports FROM reports
            """)
            remaining_reports = cursor.fetchone()[0]
        
        if 'report_sections' in existing_tables:
            cursor = conn.execute("""
                SELECT COUNT(*) as total_sections FROM report_sections
            """)
            remaining_sections = cursor.fetchone()[0]
        
        print(f"\n✅ Cleanup completed!")
        print(f"\n=== FINAL DATABASE STATS ===")
        print(f"Remaining patients: {remaining_patients:,}")
        print(f"Remaining reports: {remaining_reports:,}")
        print(f"Remaining sections: {remaining_sections:,}")
        print(f"\nRemoved:")
        print(f"  - {len(patients_to_remove):,} patients")
        print(f"  - {report_count:,} reports")
        print(f"  - {section_count:,} sections")
        
        # Verify no orphaned data
        cursor = conn.execute("""
            SELECT COUNT(*) FROM patients 
            WHERE fullname IS NULL OR fullname = '' OR fullname = ' '
        """)
        orphaned_patients = cursor.fetchone()[0]
        
        if orphaned_patients == 0:
            print("\n✅ Database is now clean - no patients without names remain.")
        else:
            print(f"\n⚠️  Warning: {orphaned_patients} patients without names still remain.")
        
        # Show some sample remaining patients
        cursor = conn.execute("""
            SELECT patient_id, fullname FROM patients 
            ORDER BY fullname 
            LIMIT 10
        """)
        
        print(f"\nSample remaining patients:")
        for row in cursor.fetchall():
            print(f"  - {row[0]}: {row[1]}")
        
    finally:
        conn.close()
    
    print(f"\n🎉 Database cleanup complete: {db_path}")

def show_patient_stats(db_path: str):
    """Show detailed statistics about patients in the database"""
    conn = sqlite3.connect(db_path)
    
    try:
        # Check what tables exist first
        existing_tables = get_existing_tables(conn)
        print(f"\nExisting tables: {', '.join(existing_tables)}")
        
        print(f"\n=== DETAILED PATIENT ANALYSIS ===")
        
        # Overall stats
        cursor = conn.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN fullname IS NOT NULL AND fullname != '' AND fullname != ' ' THEN 1 END) as with_names,
                COUNT(CASE WHEN fullname IS NULL OR fullname = '' OR fullname = ' ' THEN 1 END) as without_names,
                COUNT(CASE WHEN firstname IS NOT NULL AND firstname != '' THEN 1 END) as with_firstname,
                COUNT(CASE WHEN lastname IS NOT NULL AND lastname != '' THEN 1 END) as with_lastname
            FROM patients
        """)
        
        stats = cursor.fetchone()
        print(f"Total patients: {stats[0]:,}")
        print(f"With full names: {stats[1]:,}")
        print(f"Without full names: {stats[2]:,}")
        print(f"With first names: {stats[3]:,}")
        print(f"With last names: {stats[4]:,}")
        
        # Show examples of patients without names
        cursor = conn.execute("""
            SELECT patient_id, firstname, lastname, fullname 
            FROM patients 
            WHERE fullname IS NULL OR fullname = '' OR fullname = ' '
            LIMIT 10
        """)
        
        print(f"\nSample patients without names:")
        for row in cursor.fetchall():
            print(f"  - {row[0]}: firstname='{row[1]}', lastname='{row[2]}', fullname='{row[3]}'")
        
        # Show table sizes
        if 'reports' in existing_tables:
            cursor = conn.execute("SELECT COUNT(*) FROM reports")
            report_count = cursor.fetchone()[0]
            print(f"\nTotal reports: {report_count:,}")
        
        if 'report_sections' in existing_tables:
            cursor = conn.execute("SELECT COUNT(*) FROM report_sections")
            section_count = cursor.fetchone()[0]
            print(f"Total sections: {section_count:,}")
        
        if 'sections_fts' in existing_tables:
            cursor = conn.execute("SELECT COUNT(*) FROM sections_fts")
            fts_count = cursor.fetchone()[0]
            print(f"Sections FTS entries: {fts_count:,}")
        
        if 'reports_fts' in existing_tables:
            cursor = conn.execute("SELECT COUNT(*) FROM reports_fts")
            fts_count = cursor.fetchone()[0]
            print(f"Reports FTS entries: {fts_count:,}")
            
    finally:
        conn.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test6.py <db_path> [--stats-only]")
        print("Example: python test6.py database/base_database.sqlite")
        print("         python test6.py database/base_database.sqlite --stats-only")
        sys.exit(1)
    
    db_path = sys.argv[1]
    
    if len(sys.argv) > 2 and sys.argv[2] == '--stats-only':
        show_patient_stats(db_path)
    else:
        show_patient_stats(db_path)
        cleanup_database(db_path)