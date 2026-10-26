# DeepScan AI: Neural Media Forensics & Deepfake Intelligence Platform

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Flask 3.1](https://img.shields.io/badge/Flask-3.1-black.svg)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0-green.svg)](https://opencv.org/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind-CSS-38bdf8.svg)](https://tailwindcss.com/)

A world-class, multi-page web application and forensic intelligence platform for detecting deepfakes, synthetic faces, and manipulated media in videos and still images. Powered by a hybrid pipeline fusing **PyTorch Deep Residual Networks (ResNet-18)**, **2D Fast Fourier Transform (FFT) Frequency Domain Spectrum Analysis**, **Error Level Analysis (ELA)**, and **Cryptographic SHA-256 Chain of Custody**.

---

## 🌟 Key Features & Forensic Studios

### 1. 🎛️ Executive Mission Control (`/`)
- **Real-Time Forensic Statistics**: Total scans, deepfakes intercepted, verified authentic media, and model parameter counts.
- **Unified Quick Ingestion Dropzone**: Drag-and-drop video or photo media with dynamic laser scanning animation.
- **1-Click Preloaded Samples**: Instant evaluation of authentic speeches, synthetic face-swaps, and manipulated interview videos without needing your own files.
- **Forensic Keyframe Strip**: Instant preview of extracted keyframes with confidence ratings.

### 2. 🎬 Video Deepfake Inspector Studio (`/detect`)
- **Temporal Trajectory Graph (Chart.js)**: Frame-by-frame manipulation probability timeline exposing facial flicker and temporal inconsistencies.
- **Multi-Modal Frame Inspector**: Switch between **Facial HUD Bounding Boxes**, **2D-FFT Frequency Magnitude Spectra**, and **Attention Heatmaps**.
- **Interactive Lightbox Modal**: High-resolution deep inspection of individual frames with blur scores, spectral anomaly ratios, and millisecond timestamps.
- **Export Official Forensic PDF Certificate**: Print-optimized formal authenticity report with SHA-256 hash, timestamps, and findings.
- **JSON Evidence Export**: Complete machine-readable cryptographic audit trail.

### 3. 🖼️ Still Image & FaceSwap Forensics (`/image-forensics`)
- **Four-Way Multi-Spectral Visualizer**: Inspect spatial features, Error Level Analysis (ELA) resave differentials, 2D-FFT periodic lattice artifacts, and thermal attention gradients side-by-side.
- **Facial Landmark & Corneal Reflection Check**: Evaluates eye reflections, facial symmetry, and compression discrepancies across image boundaries.
- **EXIF & Hash Provenance**: Full cryptographic chain-of-custody tracking.

### 4. 📹 Real-Time Live Camera Scanner (`/webcam`)
- **Live Stream Verification**: Real-time webcam feed analysis running at adaptive high-frequency rates.
- **HUD Face Tracking**: Real-time bounding brackets with continuous manipulation risk score overlay.
- **Snap & Deep Inspect**: Freeze any frame from the live call and launch comprehensive multi-spectral deconstruction.
- **100% Client-Side Privacy**: Runs locally on localhost without transmitting video feeds to external cloud services.

### 5. 🗂️ Enterprise Batch & Bulk Queue (`/batch`)
- **Bulk Queue Ingestion**: Queue dozens of videos and images simultaneously.
- **Real-Time Progress Tracking**: Status monitor with per-file manipulation score and risk badges.
- **Audit CSV & JSON Export**: Download executive compliance spreadsheets for enterprise media verification.

### 6. 📜 Cryptographic Audit Log & Archive (`/history`)
- **Persistent SQLite Ledger (`forensics.db`)**: Every scan is recorded with its unique SHA-256 hash, timestamp, verdict, and raw metrics.
- **Search & Filter**: Filter by verdict (Authentic, Deepfake, Suspicious) or media type (Video, Image).
- **Inspect Past Reports**: Reopen complete forensic diagnostic reports from any historical scan.

### 7. 🧠 Neural Architecture & AI Lab (`/model-info`)
- **Interactive Topology Visualizer**: Ingestion -> Face ROI Alignment -> ResNet-18 Backbone -> 2D-FFT & ELA Fusion.
- **Benchmark Evaluation Matrix**: Comparative validation on FaceForensics++ (c23), Celeb-DF (v2), and DFDC.
- **Interactive Threshold Simulator**: Move the operational decision boundary (0.10 to 0.90) and watch simulated Precision, Recall, and Confusion Matrix (TP, FP, FN, TN) adapt in real time.

### 8. 🎓 Forensics Academy & Spot-the-Fake Challenge (`/about`)
- **Interactive 4-Round Quiz**: Test human perception against neural AI generation with instant forensic explanations.
- **The Science of Deepfakes**: Educational guides on frequency domain lattices, corneal reflections, and JPEG quantization table differences.

---

## 🏛️ Project Structure

```text
DeepfakeDetection/
│
├── app.py                            # Multi-page Flask backend & REST API
├── detector.py                       # PyTorch ResNet-18, 2D-FFT, ELA, and HUD engine
├── database.py                       # SQLite persistent audit log manager (forensics.db)
├── train.py                          # PyTorch model training pipeline
├── create_mini_dataset.py            # Dataset generator from LFW faces
├── requirements.txt                  # Python dependencies
├── haarcascade_frontalface_default.xml # Haar facial cascade
├── haarcascade_eye.xml               # Eye feature cascade
│
├── templates/                        # World-class cyber-forensic UI (Tailwind CSS)
│   ├── base.html                     # Global master layout, navigation & toasts
│   ├── index.html                    # Mission Control Dashboard
│   ├── detect.html                   # Video Deepfake Inspector Studio
│   ├── image_forensics.html          # Still Image & FaceSwap Studio
│   ├── webcam.html                   # Real-Time Live Camera Scanner
│   ├── batch.html                    # Enterprise Batch Queue Scanner
│   ├── history.html                  # Cryptographic Audit Log Archive
│   ├── model_info.html               # Model Architecture & Threshold Lab
│   └── about.html                    # Forensics Academy & Interactive Quiz
│
├── uploads/                          # Staging folder for ingested media
├── static/
│   ├── frames/                       # Generated HUD frames, 2D-FFT spectra & ELA maps
│   └── samples/                      # Preloaded authentic & deepfake test media
```

---

## 🚀 Setup & Launch (Localhost)

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python app.py
```

### 4. Access DeepScan AI
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🔬 Scientific & Forensic Methodologies

1. **2D Fast Fourier Transform (FFT)**: Generative models (StyleGAN, Diffusion, Autoencoders) use upsampling and transposed convolutions that produce high-frequency periodic lattice patterns. DeepScan computes the 2D Fourier magnitude spectrum and evaluates radial high-frequency power ratios.
2. **Error Level Analysis (ELA)**: Re-saving an image at 90% JPEG quality isolates differences in compression levels between manipulated face patches and authentic background footage.
3. **Spatial Feature ResNet Backbone**: A deep convolutional neural network extracts spatial gradients to detect blending artifacts along jaw and hairline boundaries.
4. **Cryptographic SHA-256 Digest**: Each analyzed media file is hashed to produce an immutable 64-character hexadecimal fingerprint ensuring complete chain of custody for legal and journalistic provenance.

---

## 🛡️ License & Ethics
DeepScan AI is intended for academic research, journalistic verification, digital forensics, and media authentication purposes.
