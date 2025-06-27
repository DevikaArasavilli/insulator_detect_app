from flask import Flask, request, jsonify, render_template, url_for
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import traceback

app = Flask(__name__)

# Config
upload_path = 'static/uploads'
if os.path.exists(upload_path):
    if not os.path.isdir(upload_path):
        os.remove(upload_path)
os.makedirs(upload_path, exist_ok=True)

app.config['UPLOAD_FOLDER'] = upload_path
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Load model
MODEL_PATH = 'insulator_model.h5'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Helpers
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB').resize((32, 32))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        return np.expand_dims(img_array, axis=0) / 255.0
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        raise

def detect_defects(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("cv2.imread returned None.")
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
    except Exception as e:
        print(f"❌ Defect detection error: {e}")
        return []

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            raise RuntimeError("Model not loaded.")

        if not all(view in request.files for view in ['front', 'back', 'left', 'right']):
            return jsonify({'error': 'All 4 view images are required.'}), 400

        results = []
        combined_images = []

        for view in ['front', 'back', 'left', 'right']:
            file = request.files[view]
            if file and allowed_file(file.filename):
                filename = secure_filename(f"{view}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                processed_img = preprocess_image(filepath)
                prediction = model.predict(processed_img, verbose=0)[0]
                print(f"✅ Raw prediction for {view}: {prediction}")

                # Flexible prediction shape handling
                if isinstance(prediction, np.ndarray):
                    if len(prediction) > 1:
                        confidence = float(prediction[1]) * 100
                        is_defective = prediction[1] > 0.5
                    else:
                        confidence = float(prediction[0]) * 100
                        is_defective = prediction[0] > 0.5
                else:
                    confidence = float(prediction) * 100
                    is_defective = prediction > 0.5

                print(f"View: {view}, Confidence: {confidence:.2f}%, Status: {'Defective' if is_defective else 'Normal'}")

                defects = detect_defects(filepath) if is_defective else []
                vis_path = filepath

                if is_defective:
                    img = cv2.imread(filepath)
                    if img is not None:
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

                img_vis = cv2.imread(vis_path)
                if img_vis is not None:
                    img_resized = cv2.resize(img_vis, (150, 150))
                    combined_images.append(img_resized)

        if combined_images:
            try:
                combined_image = cv2.hconcat(combined_images)
                combined_path = os.path.join(app.config['UPLOAD_FOLDER'], 'combined_visualization.jpg')
                cv2.imwrite(combined_path, combined_image)
                combined_url = url_for('static', filename='uploads/combined_visualization.jpg')
            except Exception as e:
                print(f"❌ Error combining images: {e}")
                combined_url = ""
        else:
            combined_url = ""

        overall_status = 'Defective' if any(r['prediction'] == 'Defective' for r in results) else 'Normal'

        return jsonify({
            'views': results,
            'overall_status': overall_status,
            'combined_visualization': combined_url,
            'defect_locations': [d for r in results for d in r['defects']]
        })

    except Exception as e:
        print("❌ Error during /predict route:", str(e))
        traceback.print_exc()
        return jsonify({'error': 'Error during analysis. Please try again.'}), 500

# Run the app (for local testing)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
