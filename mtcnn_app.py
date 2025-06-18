import os
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
from mtcnn import MTCNN
import numpy as np
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize MTCNN model once at startup
logger.info("Initializing global MTCNN model...")
global_detector = MTCNN()
logger.info("MTCNN model loaded successfully.")

# Function to clean up old files from uploads folder
def cleanup_old_files(directory, max_age_seconds=3600):
    """
    Remove old files from the uploads directory to prevent disk space issues.
    This could be called periodically or through a scheduled task.
    """
    import time
    current_time = time.time()
    file_count = 0
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            # If file is older than max_age_seconds, remove it
            if current_time - os.path.getmtime(file_path) > max_age_seconds:
                os.remove(file_path)
                file_count += 1
    if file_count > 0:
        logger.info(f"Cleaned up {file_count} old files from {directory}")

def count_people_mtcnn(image, confidence_threshold=0.83):
    """
    Counts people using MTCNN with a data-driven, optimized confidence threshold.
    
    - Excludes false positives below threshold
    - Includes difficult real faces above threshold
    """
    detector = MTCNN()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    all_faces = detector.detect_faces(image_rgb)
    
    # Filter based on confidence threshold
    final_faces = [face for face in all_faces if face['confidence'] >= confidence_threshold]
    
    # Draw boxes on the image for debugging
    for face in final_faces:
        x, y, width, height = face['box']
        confidence = face['confidence']
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        label = f"{confidence:.2f}"
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return len(final_faces), image

@app.route('/')
def index():
    return render_template("mtcnn_index.html")

@app.route('/mtcnn')
def mtcnn_index():
    return render_template("mtcnn_index.html")

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Check if request.json exists (image URL case)
        if request.is_json:
            data = request.get_json()
            if 'image_url' not in data:
                return jsonify({"error": "No image URL provided."}), 400
            
            image_url = data['image_url']
            # Adding browser user-agent header to fetch images
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                                    'Chrome/85.0.4183.102 Safari/537.36'}
            response = requests.get(image_url, headers=headers)
            if response.status_code != 200:
                return jsonify({"error": "Failed to fetch image from URL."}), 400
            
            # Convert image to a numpy array and decode using OpenCV
            image_array = np.frombuffer(response.content, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                return jsonify({"error": "Invalid image format."}), 400
            
            # Generate filename from URL
            filename = f"url_image_{hash(image_url) & 0xffffffff}.jpg"
        else:
            # Handle file upload
            if 'file' not in request.files:
                return jsonify({"error": "No file provided."}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No selected file."}), 400
            
            # Read file contents and decode the image
            image_array = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                return jsonify({"error": "Invalid image format."}), 400
            
            filename = file.filename
        
        # Get the confidence threshold from the request or use default
        confidence_threshold = request.form.get('confidence', 0.83)
        try:
            confidence_threshold = float(confidence_threshold)
        except (ValueError, TypeError):
            confidence_threshold = 0.83
        
        # Process the image with MTCNN
        people_count, debug_image = count_people_mtcnn(image, confidence_threshold)
        
        logger.info(f"Detected {people_count} persons in image using MTCNN")
        
        debug_filename = "mtcnn_" + filename
        debug_filepath = os.path.join(app.config['UPLOAD_FOLDER'], debug_filename)
        cv2.imwrite(debug_filepath, debug_image)
        
        return jsonify({
            "people_count": people_count,
            "debug_image": "/uploads/" + debug_filename,
            "confidence_threshold": confidence_threshold
        })
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": f"An error occurred while processing the image: {str(e)}"}), 500

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Clean up any old files at startup
    cleanup_old_files(UPLOAD_FOLDER)
    
    # Set debug mode based on environment
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    # Run the app
    app.run(debug=debug_mode, port=5000)  # Changed to default port 