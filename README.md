# PhotoCountAI - Face Detection

A web application that uses MTCNN to detect and count faces in images.

## Setup
```bash
# Clone repository
git clone https://github.com/YashasXworkz/facecountAI.git
cd facecountAI

# Setup Python 3.11 environment
# Make sure Python 3.11 is installed
python3.11 -m venv venv  # Explicitly create Python 3.11 venv
# Or on Windows: py -3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
python mtcnn_app.py
```

Access the application at `http://127.0.0.1:5000`

## Features

### MTCNN Face Detection
- Counts faces in images using MTCNN (Multi-task Cascaded Convolutional Networks)
- Adjustable confidence threshold slider (0.5-0.99)
- Well-suited for frontal-facing portraits
- Performance optimized with pre-loaded model

## How It Works
1. Upload an image using the interface
2. Adjust the confidence threshold as needed (default: 0.83)
3. The application uses MTCNN to detect faces
4. Faces with confidence scores above the threshold are counted
5. Results are displayed with bounding boxes and confidence scores

## Tech Stack
- Backend: Flask
- Face Detection: MTCNN
- Image Processing: OpenCV
- Frontend: HTML/CSS/JS

## Requirements
- Python 3.11 recommended (3.7+ supported)
- TensorFlow (for MTCNN) 