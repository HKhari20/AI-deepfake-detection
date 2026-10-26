import os
import shutil
import time
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename

import detector
import database

app = Flask(__name__)
app.config['SECRET_KEY'] = 'deepscan-cyberforensics-key-2026'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['FRAMES_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'frames')
app.config['SAMPLES_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'samples')
app.config['MAX_CONTENT_LENGTH'] = 150 * 1024 * 1024  # 150 MB limit

# Ensure required directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['FRAMES_FOLDER'], exist_ok=True)
os.makedirs(app.config['SAMPLES_FOLDER'], exist_ok=True)

# Initialize Database
database.init_db()

ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'bmp'}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# ----------------- PAGE ROUTES -----------------

@app.route('/')
def index():
    stats = database.get_stats()
    diagnostics = detector.get_system_diagnostics()
    recent_scans = database.get_all_scans(limit=5)
    return render_template('index.html', stats=stats, diagnostics=diagnostics, recent_scans=recent_scans)

@app.route('/detect')
def detect_page():
    diagnostics = detector.get_system_diagnostics()
    return render_template('detect.html', diagnostics=diagnostics)

@app.route('/image-forensics')
def image_forensics_page():
    diagnostics = detector.get_system_diagnostics()
    return render_template('image_forensics.html', diagnostics=diagnostics)

@app.route('/webcam')
def webcam_page():
    return render_template('webcam.html')

@app.route('/batch')
def batch_page():
    return render_template('batch.html')

@app.route('/history')
def history_page():
    scans = database.get_all_scans(limit=100)
    stats = database.get_stats()
    return render_template('history.html', scans=scans, stats=stats)

@app.route('/model-info')
def model_info_page():
    diagnostics = detector.get_system_diagnostics()
    return render_template('model_info.html', diagnostics=diagnostics)

@app.route('/about')
def about_page():
    return render_template('about.html')

# ----------------- API ENDPOINTS -----------------

@app.route('/api/analyze-video', methods=['POST'])
def api_analyze_video():
    try:
        sample_name = request.form.get('sample_name')
        sensitivity = float(request.form.get('sensitivity', 0.5))
        sample_count = int(request.form.get('sample_count', 12))
        filepath = None

        if sample_name:
            # Use preloaded sample
            sample_path = os.path.join(app.config['SAMPLES_FOLDER'], secure_filename(sample_name))
            if os.path.exists(sample_path):
                filepath = sample_path
            else:
                return jsonify({"error": f"Sample file not found: {sample_name}"}), 404
        else:
            if 'video' not in request.files:
                return jsonify({"error": "No video file provided"}), 400
            file = request.files['video']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
                return jsonify({"error": "Unsupported video format. Allowed: MP4, AVI, MOV, MKV, WEBM"}), 400

            filename = secure_filename(file.filename)
            unique_filename = f"{int(time.time())}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)

        # Run Video Deepfake Detection
        results = detector.process_video(
            filepath,
            output_dir=app.config['FRAMES_FOLDER'],
            sample_count=sample_count,
            sensitivity_threshold=sensitivity
        )

        # Save to SQLite Database
        database.save_scan(results)

        return jsonify(results)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze-image', methods=['POST'])
def api_analyze_image():
    try:
        sample_name = request.form.get('sample_name')
        sensitivity = float(request.form.get('sensitivity', 0.5))
        filepath = None

        if sample_name:
            sample_path = os.path.join(app.config['SAMPLES_FOLDER'], secure_filename(sample_name))
            if os.path.exists(sample_path):
                filepath = sample_path
            else:
                return jsonify({"error": f"Sample file not found: {sample_name}"}), 404
        else:
            if 'image' not in request.files:
                return jsonify({"error": "No image file provided"}), 400
            file = request.files['image']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
                return jsonify({"error": "Unsupported image format. Allowed: JPG, PNG, WEBP, BMP"}), 400

            filename = secure_filename(file.filename)
            unique_filename = f"{int(time.time())}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)

        # Run Image Deepfake Forensics
        results = detector.process_image(
            filepath,
            output_dir=app.config['FRAMES_FOLDER'],
            sensitivity_threshold=sensitivity
        )

        # Save to SQLite Database
        database.save_scan(results)

        return jsonify(results)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze-frame', methods=['POST'])
def api_analyze_frame():
    """Receives a base64 encoded frame from webcam stream for real-time inference."""
    try:
        data = request.get_json(force=True)
        image_b64 = data.get('image')
        if not image_b64:
            return jsonify({"error": "No image data provided"}), 400

        result = detector.process_frame_base64(image_b64)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/history', methods=['GET'])
def api_get_history():
    scans = database.get_all_scans(limit=100)
    return jsonify(scans)

@app.route('/api/history/<scan_id>', methods=['GET'])
def api_get_scan_detail(scan_id):
    scan = database.get_scan_by_id(scan_id)
    if scan:
        return jsonify(scan)
    return jsonify({"error": "Scan record not found"}), 404

@app.route('/api/history/<scan_id>', methods=['DELETE'])
def api_delete_scan(scan_id):
    success = database.delete_scan(scan_id)
    return jsonify({"success": success})

@app.route('/api/history/clear', methods=['POST'])
def api_clear_history():
    database.clear_all_scans()
    return jsonify({"success": True})

@app.route('/api/samples', methods=['GET'])
def api_get_samples():
    samples = []
    if os.path.exists(app.config['SAMPLES_FOLDER']):
        for fname in os.listdir(app.config['SAMPLES_FOLDER']):
            fpath = os.path.join(app.config['SAMPLES_FOLDER'], fname)
            if os.path.isfile(fpath):
                ext = fname.rsplit('.', 1)[-1].lower()
                is_video = ext in ALLOWED_VIDEO_EXTENSIONS
                is_image = ext in ALLOWED_IMAGE_EXTENSIONS
                if is_video or is_image:
                    samples.append({
                        "name": fname,
                        "type": "video" if is_video else "image",
                        "size_mb": round(os.path.getsize(fpath) / (1024 * 1024), 2),
                        "url": f"/static/samples/{fname}"
                    })
    return jsonify(samples)

@app.route('/api/stats', methods=['GET'])
def api_get_stats():
    stats = database.get_stats()
    diagnostics = detector.get_system_diagnostics()
    return jsonify({"stats": stats, "diagnostics": diagnostics})

if __name__ == '__main__':
    print("===============================================================")
    print("   DEEPSCAN AI - NEURAL MEDIA FORENSICS INTELLIGENCE PLATFORM  ")
    print("   Running locally on: http://127.0.0.1:5000                   ")
    print("===============================================================")
    app.run(debug=True, host='127.0.0.1', port=5000)
