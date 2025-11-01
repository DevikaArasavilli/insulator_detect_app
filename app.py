from flask import Flask, request, jsonify, render_template, url_for, flash, redirect
import os, glob
from werkzeug.utils import secure_filename
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import traceback

app = Flask(__name__)
app.secret_key = "supersecretkey123"  # Required for flash messages

# ---------------- CONFIG ---------------- #
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Clear previous uploads
for f in glob.glob(os.path.join(UPLOAD_FOLDER, "*")):
    if os.path.isfile(f):
        try:
            os.remove(f)
        except:
            pass

# ---------------- LOAD MODEL ---------------- #
# ✅ Corrected model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'insulator_model.h5')

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# ---------------- HELPERS ---------------- #
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except:
        return None

def detect_defects(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    defects = []
    for c in contours:
        area = cv2.contourArea(c)
        if 100 < area < 5000:
            x, y, w, h = cv2.boundingRect(c)
            defects.append({'x': x, 'y': y, 'w': w, 'h': h, 'severity': 'high' if area > 1000 else 'medium'})
    return defects

# ---------------- ROUTES ---------------- #
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# ---------------- CONTACT FORM HANDLER ---------------- #
@app.route('/send_message', methods=['POST'])
def send_message():
    try:
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        print(f"New message from {name} ({email}): {message}")

        flash("Your message has been sent successfully!", "success")
        return redirect(url_for('contact'))

    except Exception as e:
        print(f"Error sending message: {e}")
        flash("There was an error sending your message. Please try again.", "error")
        return redirect(url_for('contact'))

# ---------------- PREDICTION ROUTE ---------------- #
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        required_views = ['front', 'back', 'left', 'right']
        if not all(view in request.files for view in required_views):
            return jsonify({'error': 'All 4 views required'}), 400

        results = []
        for view in required_views:
            file = request.files[view]
            if file and allowed_file(file.filename):
                filename = secure_filename(f"{view}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                processed = preprocess_image(filepath)
                if processed is None:
                    continue

                prediction = model.predict(processed, verbose=0)[0]
                confidence = float(prediction[0]) * 100
                is_defective = prediction[0] > 0.5
                defects = detect_defects(filepath) if is_defective else []

                vis_path = filepath
                if is_defective and defects:
                    img = cv2.imread(filepath)
                    for d in defects:
                        cv2.rectangle(img, (d['x'], d['y']), (d['x'] + d['w'], d['y'] + d['h']), (0, 0, 255), 2)
                    vis_path = os.path.join(app.config['UPLOAD_FOLDER'], f"vis_{filename}")
                    cv2.imwrite(vis_path, img)

                results.append({
                    'view': view.capitalize(),
                    'prediction': 'Defective' if is_defective else 'Normal',
                    'confidence': round(confidence, 2),
                    'visualization': url_for('static', filename=f"uploads/{os.path.basename(vis_path)}"),
                    'reason': "Detected defects" if is_defective else "No visible defects"
                })

        return jsonify({'results': results})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'Error during analysis'}), 500

# ---------------- MAIN ---------------- #
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
