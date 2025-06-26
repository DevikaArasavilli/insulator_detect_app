from flask import Flask, request, jsonify, render_template, url_for
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# Initialize app
app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Load the model
MODEL_PATH = 'insulator_model.h5'
model = tf.keras.models.load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# Utility functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB').resize((32, 32))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    return np.expand_dims(img_array, axis=0) / 255.0

def detect_defects(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    defects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 5000:
            x, y, w, h = cv2.boundingRect(contour)
            defects.append({'x': x, 'y': y, 'severity': 'high' if area > 1000 else 'medium'})
    return defects

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded properly'}), 500

    if not all(view in request.files for view in ['front', 'back', 'left', 'right']):
        return jsonify({'error': 'All 4 view images (front, back, left, right) are required.'}), 400

    results = []
    combined_images = []

    for view in ['front', 'back', 'left', 'right']:
        file = request.files[view]
        if file and allowed_file(file.filename):
            filename = secure_filename(f"{view}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            processed_img = preprocess_image(filepath)
            prediction = model.predict(processed_img)[0]
            confidence = float(prediction[1]) * 100
            is_defective = prediction[1] > 0.5

            defects = detect_defects(filepath) if is_defective else []

            vis_path = filepath
            if is_defective:
                img = cv2.imread(filepath)
                for defect in defects:
                    cv2.rectangle(img, (defect['x'], defect['y']),
                                  (defect['x'] + 10, defect['y'] + 10), (0, 0, 255), 2)
                vis_path = os.path.join(app.config['UPLOAD_FOLDER'], f"vis_{filename}")
                cv2.imwrite(vis_path, img)

            results.append({
                'view': view.capitalize(),
                'prediction': 'Defective' if is_defective else 'Normal',
                'confidence': round(confidence, 2),
                'visualization': url_for('static', filename=f"uploads/{os.path.basename(vis_path)}"),
                'defects': defects
            })

            # For combined image grid
            combined_images.append(cv2.resize(cv2.imread(vis_path), (150, 150)))

    # Create combined visualization
    if combined_images:
        combined_image = cv2.hconcat(combined_images)
        combined_path = os.path.join(app.config['UPLOAD_FOLDER'], 'combined_visualization.jpg')
        cv2.imwrite(combined_path, combined_image)
        combined_url = url_for('static', filename='uploads/combined_visualization.jpg')
    else:
        combined_url = ""

    overall_status = 'Defective' if any(r['prediction'] == 'Defective' for r in results) else 'Normal'

    return jsonify({
        'views': results,
        'overall_status': overall_status,
        'combined_visualization': combined_url,
        'defect_locations': [d for r in results for d in r['defects']]
    })

# Needed for Render deployment
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
