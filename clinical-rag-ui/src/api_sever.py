from flask import Flask, jsonify, request
from flask_cors import CORS
import sqlite3
import json
import re
import httpx
import os
import csv
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for React app

class ClinicalRAGDB:
    def __init__(self, db_path: str = "database/base_database.sqlite"):
        self.db_path = db_path
    
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        return conn
    
    def search_patients(self, query: str, limit: int = 50):
        """Search patients by name, patient ID, or DOB with support for German DOB format"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if query.strip():
            # Search in name fields, patient_id, and DOB (including German format conversion)
            cursor.execute('''
                SELECT patient_id, firstname, lastname, fullname, dob 
                FROM patients 
                WHERE fullname LIKE ? 
                   OR firstname LIKE ? 
                   OR lastname LIKE ? 
                   OR patient_id LIKE ?
                   OR dob LIKE ?
                   OR (dob IS NOT NULL AND 
                       CASE 
                           WHEN dob LIKE '____-__-__' THEN 
                               strftime('%d.%m.%Y', dob) LIKE ?
                           ELSE 
                               dob LIKE ?
                       END)
                ORDER BY 
                    CASE 
                        WHEN patient_id LIKE ? THEN 1
                        WHEN fullname LIKE ? THEN 2
                        WHEN firstname LIKE ? OR lastname LIKE ? THEN 3
                        ELSE 4
                    END,
                    fullname 
                LIMIT ?
            ''', (
                f'%{query}%',  # fullname
                f'%{query}%',  # firstname  
                f'%{query}%',  # lastname
                f'%{query}%',  # patient_id
                f'%{query}%',  # dob direct match
                f'%{query}%',  # German format DOB conversion
                f'%{query}%',  # German format DOB fallback
                f'{query}%',   # patient_id exact match (for ordering)
                f'{query}%',   # fullname exact match (for ordering)
                f'{query}%',   # firstname exact match (for ordering)
                f'{query}%',   # lastname exact match (for ordering)
                limit
            ))
        else:
            # Return all patients if no query
            cursor.execute('''
                SELECT patient_id, firstname, lastname, fullname, dob 
                FROM patients 
                ORDER BY fullname 
                LIMIT ?
            ''', (limit,))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        # Debug logging
        print(f"🔍 Patient search for '{query}' returned {len(results)} results")
        if results:
            print(f"📋 Sample result: {results[0]}")
        
        return results
    
    def get_report_types(self):
        """Get all available report types"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT report_type 
            FROM reports 
            WHERE report_type IS NOT NULL 
            ORDER BY report_type
        ''')
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def get_section_types(self):
        """Get all available section types ordered by frequency (most common first)"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT section_name, COUNT(*) as count
            FROM report_sections 
            WHERE section_name IS NOT NULL 
            GROUP BY section_name
            ORDER BY COUNT(*) DESC, section_name
        ''')
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def search_sections(self, query: str, patient_id=None, patient_ids=None, k: int = 8, 
                   report_types=None, section_types=None, 
                   start_date=None, end_date=None):
        """Search in report sections using FTS - now supports both single patient and group queries"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Escape the FTS query
        escaped_query = escape_fts_query(query)
        
        sql = '''
            SELECT 
                rs.section_id,
                rs.report_id,
                rs.patient_id,
                rs.section_name,
                rs.section_content,
                rs.section_order,
                rs.word_count,
                rs.filename,
                rs.report_type,
                rs.report_date,
                p.fullname as patient_name,
                snippet(sections_fts, 6, '<mark>', '</mark>', '...', 200) as snippet,
                bm25(sections_fts) as score
            FROM sections_fts fts
            JOIN report_sections rs ON rs.section_id = fts.section_id
            LEFT JOIN patients p ON p.patient_id = rs.patient_id
            WHERE sections_fts MATCH ?
        '''
        
        params = [escaped_query]
        
        # Handle both single patient and group queries
        if patient_id:
            sql += ' AND rs.patient_id = ?'
            params.append(patient_id)
        elif patient_ids:
            placeholders = ','.join(['?' for _ in patient_ids])
            sql += f' AND rs.patient_id IN ({placeholders})'
            params.extend(patient_ids)
        
        if report_types:
            placeholders = ','.join(['?' for _ in report_types])
            sql += f' AND rs.report_type IN ({placeholders})'
            params.extend(report_types)
        
        if section_types:
            placeholders = ','.join(['?' for _ in section_types])
            sql += f' AND rs.section_name IN ({placeholders})'
            params.extend(section_types)
        
        if start_date:
            sql += ' AND rs.report_date >= ?'
            params.append(start_date)
        
        if end_date:
            sql += ' AND rs.report_date <= ?'
            params.append(end_date)
        
        sql += ' ORDER BY bm25(sections_fts) DESC LIMIT ?'
        params.append(k)
        
        try:
            cursor.execute(sql, params)
            
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                
                # Clean up snippet
                if result['snippet']:
                    result['snippet'] = result['snippet'].replace('<mark>', '').replace('</mark>', '')
                else:
                    result['snippet'] = result['section_content'][:200] + '...'
                
                # Make score positive
                result['score'] = abs(float(result['score'])) if result['score'] else 0.0
                
                results.append(result)
            
            conn.close()
            return results
            
        except Exception as e:
            print(f"❌ Sections SQL Error: {e}")
            conn.close()
            return []
    
    def search_reports(self, query: str, patient_id=None, patient_ids=None, k: int = 8, 
                  report_types=None, start_date=None, end_date=None):
        """Search in full reports using FTS - now supports both single patient and group queries"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Escape the FTS query
        escaped_query = escape_fts_query(query)
        
        sql = '''
            SELECT 
                r.report_id,
                r.patient_id,
                r.filename,
                r.report_type,
                r.report_date,
                r.created_at,
                r.source_path,
                p.fullname as patient_name,
                snippet(reports_fts, 2, '<mark>', '</mark>', '...', 200) as snippet,
                bm25(reports_fts) as score,
                length(r.content) as word_count,
                r.content
            FROM reports_fts fts
            JOIN reports r ON r.report_id = fts.report_id
            LEFT JOIN patients p ON p.patient_id = r.patient_id
            WHERE reports_fts MATCH ?
        '''
        
        params = [escaped_query]
        
        # Handle both single patient and group queries
        if patient_id:
            sql += ' AND r.patient_id = ?'
            params.append(patient_id)
        elif patient_ids:
            placeholders = ','.join(['?' for _ in patient_ids])
            sql += f' AND r.patient_id IN ({placeholders})'
            params.extend(patient_ids)
        
        if report_types:
            placeholders = ','.join(['?' for _ in report_types])
            sql += f' AND r.report_type IN ({placeholders})'
            params.extend(report_types)
        
        if start_date:
            sql += ' AND r.report_date >= ?'
            params.append(start_date)
        
        if end_date:
            sql += ' AND r.report_date <= ?'
            params.append(end_date)
        
        sql += ' ORDER BY bm25(reports_fts) DESC LIMIT ?'
        params.append(k)
        
        try:
            cursor.execute(sql, params)
            
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                # Clean up snippet
                if result['snippet']:
                    result['snippet'] = result['snippet'].replace('<mark>', '').replace('</mark>', '')
                else:
                    result['snippet'] = result['content'][:200] + '...'
                
                # Make score positive
                result['score'] = abs(float(result['score'])) if result['score'] else 0.0
                
                results.append(result)
            
            conn.close()
            return results
            
        except Exception as e:
            print(f"❌ Reports SQL Error: {e}")
            conn.close()
            return []
    
    def search_combined(self, query: str, patient_id=None, patient_ids=None, k: int = 8, 
                       report_types=None, section_types=None, 
                       start_date=None, end_date=None):
        """Search both sections and reports, then combine and rank results - supports group queries"""
        # Get half results from each
        sections_k = k // 2
        reports_k = k - sections_k
        
        sections = self.search_sections(query, patient_id, patient_ids, sections_k, 
                                      report_types, section_types, start_date, end_date)
        reports = self.search_reports(query, patient_id, patient_ids, reports_k, 
                                    report_types, start_date, end_date)
        
        # Combine and sort by score
        all_results = sections + reports
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        return all_results[:k]

    def get_full_report(self, report_id: str):
        """Get full report content by report ID"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT report_id, patient_id, filename, report_type, report_date, 
                   content, created_at, source_path
            FROM reports 
            WHERE report_id = ?
        ''', (report_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return dict(result)
        return None

# Initialize database connection
db = ClinicalRAGDB()

def escape_fts_query(query):
    """Escape special characters in FTS queries"""
    # Remove or escape problematic characters
    query = re.sub(r'[\'\"*(){}[\]]', ' ', query)  # Remove quotes and special chars
    query = re.sub(r'\s+', ' ', query)  # Normalize whitespace
    query = query.strip()
    
    # Split into words and quote each one for exact matching
    words = query.split()
    if len(words) > 1:
        # For multi-word queries, use OR between words
        escaped = ' OR '.join(f'"{word}"' for word in words if len(word) > 2)
    else:
        escaped = f'"{words[0]}"' if words and len(words[0]) > 2 else query
    
    return escaped if escaped else query

@app.route('/api/patients/search', methods=['POST'])
def search_patients():
    data = request.json
    query = data.get('query', '')
    
    try:
        results = db.search_patients(query)
        return jsonify(results)
    except Exception as e:
        print(f"❌ Error in patients search: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reports/types', methods=['GET'])
def get_report_types():
    try:
        results = db.get_report_types()
        return jsonify(results)
    except Exception as e:
        print(f"❌ Error getting report types: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sections/types', methods=['GET'])
def get_section_types():
    try:
        results = db.get_section_types()
        return jsonify(results)
    except Exception as e:
        print(f"❌ Error getting section types: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sections/search', methods=['POST'])
def search_sections():
    data = request.json
    query = data.get('query', '')
    patient_id = data.get('patient_id')
    patient_ids = data.get('patient_ids')  # Add support for group queries
    k = data.get('k', 8)
    report_types = data.get('report_types')
    section_types = data.get('section_types')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    
    try:
        results = db.search_sections(query, patient_id, patient_ids, k, report_types, 
                                   section_types, start_date, end_date)
        return jsonify(results)
    except Exception as e:
        print(f"❌ Error in sections search: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reports/search', methods=['POST'])
def search_reports():
    data = request.json
    query = data.get('query', '')
    patient_id = data.get('patient_id')
    patient_ids = data.get('patient_ids')  # Add support for group queries
    k = data.get('k', 8)
    report_types = data.get('report_types')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    
    try:
        results = db.search_reports(query, patient_id, patient_ids, k, report_types, 
                                  start_date, end_date)
        return jsonify(results)
    except Exception as e:
        print(f"❌ Error in reports search: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search_combined():
    """Combined search endpoint for both sections and reports - supports group queries"""
    data = request.json
    query = data.get('query', '')
    patient_id = data.get('patient_id')
    patient_ids = data.get('patient_ids')  # Add support for group queries
    k = data.get('k', 8)
    report_types = data.get('report_types')
    section_types = data.get('section_types')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    search_level = data.get('search_level', 'both')
    
    try:
        if search_level == 'sections':
            results = db.search_sections(query, patient_id, patient_ids, k, report_types, 
                                       section_types, start_date, end_date)
        elif search_level == 'reports':
            results = db.search_reports(query, patient_id, patient_ids, k, report_types, 
                                      start_date, end_date)
        else:  # 'both'
            results = db.search_combined(query, patient_id, patient_ids, k, report_types, 
                                       section_types, start_date, end_date)
        
        return jsonify(results)
    except Exception as e:
        print(f"❌ Error in combined search: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/llama/chat', methods=['POST'])
def proxy_llama():
    """Proxy requests to the local Llama model"""
    try:
        data = request.json
        print(f"🦙 Proxying Llama request: {data.get('messages', [{}])[-1].get('content', '')[:100]}...")
        
        # Forward to local Llama server (this runs on the cluster where the API server is)
        response = httpx.post(
            "http://localhost:9999/v1/chat/completions",
            json=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer dacbebe8c973154018a3d0f5",
            },
            timeout=120.0
        )
        
        print(f"🦙 Llama server response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"🦙 Llama response success: {len(result.get('choices', [{}])[0].get('message', {}).get('content', ''))} chars")
            return jsonify(result)
        else:
            print(f"❌ Llama server error: {response.status_code} - {response.text}")
            return jsonify({
                'error': f'Llama server error: {response.status_code}',
                'detail': response.text
            }), response.status_code
            
    except httpx.TimeoutException:
        print("❌ Llama request timeout")
        return jsonify({'error': 'Llama server timeout'}), 504
    except httpx.ConnectError:
        print("❌ Cannot connect to Llama server")
        return jsonify({'error': 'Cannot connect to Llama server at localhost:9999'}), 503
    except Exception as e:
        print(f"❌ Llama proxy error: {e}")
        return jsonify({'error': f'Proxy error: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'Clinical RAG API is running'})

@app.route('/api/reports/<report_id>', methods=['GET'])
def get_full_report_endpoint(report_id):
    try:
        print(f"📄 Fetching full report: {report_id}")
        result = db.get_full_report(report_id)
        if result:
            print(f"✅ Report found: {len(result.get('content', ''))} characters")
            return jsonify(result)
        else:
            print(f"❌ Report not found: {report_id}")
            return jsonify({'error': 'Report not found'}), 404
    except Exception as e:
        print(f"❌ Error fetching report {report_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/logging/session', methods=['POST'])
def log_session():
    """Log a clinical RAG session to file for research purposes"""
    try:
        data = request.json
        filepath = data.get('filepath')
        format_type = data.get('format', 'json')
        
        if not filepath:
            return jsonify({'error': 'filepath is required'}), 400
        
        # Handle directory-only paths by adding a filename
        if os.path.isdir(filepath) or filepath.endswith('/'):
            # Generate a filename based on current timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            extension = 'json' if format_type == 'json' else 'csv'
            filename = f"clinical_rag_log_{timestamp}.{extension}"
            filepath = os.path.join(filepath.rstrip('/'), filename)
            print(f"📝 Directory provided, using filename: {filepath}")
        
        # Validate the filepath and ensure directory exists
        try:
            # Get the directory path
            directory = os.path.dirname(os.path.abspath(filepath))
            if directory:  # Only create if there's actually a directory part
                os.makedirs(directory, exist_ok=True)
                print(f"📁 Ensured directory exists: {directory}")
            
            # Test write permissions by attempting to create/touch the file
            if not os.path.exists(filepath):
                with open(filepath, 'a', encoding='utf-8') as test_file:
                    pass  # Just create the file if it doesn't exist
                print(f"📝 Created log file: {filepath}")
                
        except PermissionError as perm_error:
            return jsonify({'error': f'Permission denied: {perm_error}. Check file/directory permissions.'}), 403
        except Exception as dir_error:
            return jsonify({'error': f'Cannot create directory or file: {dir_error}'}), 400
        
        # Remove API-specific fields from the log entry
        log_entry = {k: v for k, v in data.items() if k not in ['filepath', 'format']}
        
        print(f"📝 Logging session to: {filepath} (format: {format_type})")
        
        if format_type == 'json':
            # Append to JSON Lines file (one JSON object per line)
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                
        elif format_type == 'csv':
            # Helper function to clean text for CSV
            def clean_csv_text(text):
                if text is None:
                    return ''
                if isinstance(text, (list, dict)):
                    text = str(text)
                # Convert to string and replace newlines with spaces
                text = str(text).replace('\n', ' ').replace('\r', ' ')
                # Replace multiple spaces with single space
                text = ' '.join(text.split())
                # Truncate very long text to avoid CSV issues
                if len(text) > 1000:
                    text = text[:997] + '...'
                return text
            
            # Flatten the data for CSV format with proper text cleaning
            flat_data = {
                'timestamp': log_entry.get('timestamp', ''),
                'session_id': log_entry.get('session_id', ''),
                'patient_id': log_entry.get('patient', {}).get('id', ''),
                'patient_name': clean_csv_text(log_entry.get('patient', {}).get('name', '')),
                'patient_dob': log_entry.get('patient', {}).get('dob', ''),
                'question': clean_csv_text(log_entry.get('query', {}).get('question', '')),
                'k': log_entry.get('query', {}).get('settings', {}).get('k', ''),
                'search_level': log_entry.get('query', {}).get('settings', {}).get('searchLevel', ''),
                'doc_types': ','.join(log_entry.get('query', {}).get('settings', {}).get('docTypes', [])),
                'section_types': clean_csv_text(','.join(log_entry.get('query', {}).get('settings', {}).get('sectionTypes', []))),
                'date_start': log_entry.get('query', {}).get('settings', {}).get('dateRange', {}).get('start', ''),
                'date_end': log_entry.get('query', {}).get('settings', {}).get('dateRange', {}).get('end', ''),
                'answer_status': log_entry.get('results', {}).get('answer', {}).get('status', ''),
                'answer_text': clean_csv_text(log_entry.get('results', {}).get('answer', {}).get('text', '')),
                'source_count': log_entry.get('results', {}).get('sourceCount', ''),
                'model_id': log_entry.get('metadata', {}).get('modelId', ''),
                'retrieval_latency_ms': log_entry.get('metadata', {}).get('latency', {}).get('retrieval', ''),
                'generation_latency_ms': log_entry.get('metadata', {}).get('latency', {}).get('generation', ''),
                'total_latency_ms': log_entry.get('metadata', {}).get('latency', {}).get('total', ''),
                'index_version': log_entry.get('metadata', {}).get('indexVersion', '')
            }
            
            # Check if file exists to determine if we need to write headers
            file_exists = os.path.exists(filepath)
            
            with open(filepath, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=flat_data.keys(), quoting=csv.QUOTE_MINIMAL)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(flat_data)
        
        else:
            return jsonify({'error': f'Unsupported format: {format_type}'}), 400
        
        print(f"✅ Successfully logged session to: {filepath}")
        return jsonify({
            'status': 'success', 
            'message': f'Session logged to {filepath}',
            'format': format_type,
            'actualFilepath': filepath  # Return the actual filepath used
        })
        
    except Exception as e:
        print(f"❌ Logging error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Add a test endpoint to verify logging is working
@app.route('/api/logging/test', methods=['GET'])
def test_logging():
    """Test endpoint to verify logging functionality"""
    return jsonify({
        'status': 'ok',
        'message': 'Logging endpoint is available',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/logging/content', methods=['POST'])
def get_log_content():
    """Get the content of a log file for the live feed"""
    try:
        data = request.json
        filepath = data.get('filepath')
        format_type = data.get('format', 'json')
        
        if not filepath:
            return jsonify({'error': 'filepath is required'}), 400
        
        if not os.path.exists(filepath):
            return jsonify({'content': '', 'message': 'Log file does not exist yet'})
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return jsonify({
                'content': content,
                'format': format_type,
                'filepath': filepath,
                'file_size': len(content),
                'last_modified': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
            })
            
        except Exception as read_error:
            print(f"❌ Error reading log file {filepath}: {read_error}")
            return jsonify({'error': f'Cannot read log file: {read_error}'}), 500
        
    except Exception as e:
        print(f"❌ Error in get_log_content: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🏥 Starting Clinical RAG API Server...")
    print("Available endpoints:")
    print("  POST /api/patients/search - Search patients")
    print("  GET  /api/reports/types - Get report types")
    print("  GET  /api/sections/types - Get section types")
    print("  POST /api/sections/search - Search sections (supports groups)")
    print("  POST /api/reports/search - Search reports (supports groups)")
    print("  POST /api/search - Combined search (supports groups)")
    print("  POST /api/llama/chat - Llama model proxy")
    print("  GET  /api/reports/<report_id> - Get full report")
    print("  POST /api/logging/session - Log session data")
    print("  POST /api/logging/content - Get log file content")
    print("  GET  /api/health - Health check")
    print(f"Database: {db.db_path}")
    app.run(host='0.0.0.0', port=5200, debug=True)  # Listen on all interfaces for network access