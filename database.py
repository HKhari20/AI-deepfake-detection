import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'forensics.db')

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            media_type TEXT NOT NULL,
            file_hash TEXT,
            timestamp TEXT NOT NULL,
            fake_probability REAL NOT NULL,
            verdict TEXT NOT NULL,
            confidence REAL NOT NULL,
            fps REAL DEFAULT 0,
            resolution TEXT DEFAULT 'N/A',
            frame_count INTEGER DEFAULT 1,
            faces_found INTEGER DEFAULT 0,
            details_json TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_scan(scan_data):
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    
    scan_id = scan_data.get('scan_id', f"scan_{int(datetime.now().timestamp()*1000)}")
    filename = scan_data.get('filename', 'unknown_media')
    media_type = scan_data.get('media_type', 'video')
    file_hash = scan_data.get('file_hash', 'N/A')
    timestamp = scan_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    fake_prob = float(scan_data.get('overall_fake_probability', 0.0))
    verdict = scan_data.get('verdict', 'SUSPICIOUS')
    confidence = float(scan_data.get('confidence', abs(fake_prob - 0.5) * 200))
    
    meta = scan_data.get('metadata', {})
    fps = float(meta.get('fps', 0.0))
    resolution = str(meta.get('resolution', 'N/A'))
    frame_count = int(meta.get('frame_count', 1))
    
    faces_found = int(scan_data.get('faces_found', 0))
    if faces_found == 0 and 'frames' in scan_data:
        faces_found = sum(f.get('faces_detected', 0) for f in scan_data['frames'])
        
    details_json = json.dumps(scan_data)
    
    cursor.execute('''
        INSERT OR REPLACE INTO scans (
            scan_id, filename, media_type, file_hash, timestamp,
            fake_probability, verdict, confidence, fps, resolution,
            frame_count, faces_found, details_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        scan_id, filename, media_type, file_hash, timestamp,
        fake_prob, verdict, confidence, fps, resolution,
        frame_count, faces_found, details_json
    ))
    conn.commit()
    conn.close()
    return scan_id

def get_all_scans(limit=100):
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM scans ORDER BY id DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    scans = [dict(row) for row in rows]
    conn.close()
    return scans

def get_scan_by_id(scan_id):
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM scans WHERE scan_id = ?', (scan_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        d = dict(row)
        if d.get('details_json'):
            try:
                d['details'] = json.loads(d['details_json'])
            except Exception:
                d['details'] = {}
        return d
    return None

def delete_scan(scan_id):
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM scans WHERE scan_id = ?', (scan_id,))
    conn.commit()
    conn.close()
    return True

def clear_all_scans():
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM scans')
    conn.commit()
    conn.close()
    return True

def get_stats():
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM scans')
    total_scans = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM scans WHERE verdict LIKE '%DEEPFAKE%'")
    fake_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM scans WHERE verdict LIKE '%AUTHENTIC%'")
    authentic_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT AVG(fake_probability) FROM scans')
    avg_fake_prob = cursor.fetchone()[0] or 0.0
    
    conn.close()
    
    return {
        "total_scans": total_scans,
        "deepfakes_detected": fake_count,
        "authentic_verified": authentic_count,
        "suspicious_flagged": total_scans - (fake_count + authentic_count),
        "avg_fake_probability": round(avg_fake_prob, 3)
    }

if __name__ == '__main__':
    init_db()
    print("Database initialized successfully at", DB_PATH)
