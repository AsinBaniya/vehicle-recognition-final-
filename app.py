import os
import json
import base64
from io import BytesIO
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__, static_folder=None)

# Constants
CLASSES = ['Bus', 'Car', 'Motorcycle', 'Truck']
TARGET_SIZE = (128, 128)
PROJECT_DIR = os.path.dirname(__file__)
PUBLIC_FOLDER = os.path.join(PROJECT_DIR, 'public')
SAMPLES_FOLDER = os.path.join(PUBLIC_FOLDER, 'samples')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

# Load the trained model
MODEL_DIR = os.path.join(PROJECT_DIR, 'model')
MODEL_PATH_KERAS = os.path.join(MODEL_DIR, 'best_vehicle_model.keras')
MODEL_PATH_H5 = os.path.join(MODEL_DIR, 'best_vehicle_model.h5')

model = None

def get_model():
    global model
    if model is None:
        import tensorflow as tf
        if os.path.exists(MODEL_PATH_KERAS):
            print(f"Loading Keras model from {MODEL_PATH_KERAS}")
            model = tf.keras.models.load_model(MODEL_PATH_KERAS)
        elif os.path.exists(MODEL_PATH_H5):
            print(f"Loading H5 model from {MODEL_PATH_H5}")
            model = tf.keras.models.load_model(MODEL_PATH_H5)
        else:
            raise FileNotFoundError("Trained vehicle model not found in model/ folder.")
    return model

def preprocess_image(image_file):
    """
    Preprocess an uploaded image in memory and return model input and preview.
    """
    with Image.open(image_file) as source:
        img = source.convert('RGB')
        resized = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        arr = np.array(resized, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)  # Shape (1, 128, 128, 3)

        preview = img.copy()
        preview.thumbnail((512, 512), Image.Resampling.LANCZOS)
        preview_buffer = BytesIO()
        preview.save(preview_buffer, format='JPEG', quality=85, optimize=True)
        preview_url = (
            'data:image/jpeg;base64,'
            + base64.b64encode(preview_buffer.getvalue()).decode('ascii')
        )

    return arr, preview_url

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<path:filename>')
def public_asset(filename):
    return send_from_directory(PUBLIC_FOLDER, filename)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided in request.'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected.'}), 400
    
    try:
        processed_input, preview_url = preprocess_image(file.stream)
        trained_model = get_model()
        preds = trained_model.predict(processed_input)[0]
        
        pred_idx = int(np.argmax(preds))
        pred_class = CLASSES[pred_idx]
        pred_confidence = float(preds[pred_idx]) * 100
        
        probabilities = {
            CLASSES[i]: round(float(preds[i]) * 100, 2)
            for i in range(len(CLASSES))
        }
        
        # Vehicle descriptions / metadata
        vehicle_meta = {
            'Bus': {'type': 'Heavy Public Transit', 'icon': 'fa-bus', 'badge': 'Commercial Transport'},
            'Car': {'type': 'Passenger Vehicle', 'icon': 'fa-car-side', 'badge': 'Private Transport'},
            'Motorcycle': {'type': 'Two-Wheeled Motor Vehicle', 'icon': 'fa-motorcycle', 'badge': 'Light Transport'},
            'Truck': {'type': 'Heavy Freight & Cargo', 'icon': 'fa-truck', 'badge': 'Freight Transport'}
        }
        
        return jsonify({
            'success': True,
            'prediction': pred_class,
            'confidence': round(pred_confidence, 2),
            'probabilities': probabilities,
            'image_url': preview_url,
            'meta': vehicle_meta.get(pred_class, {})
        })
        
    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

@app.route('/api/sample/<class_name>')
def get_sample(class_name):
    sample_name = class_name.lower()
    if sample_name in {'bus', 'car', 'motorcycle', 'truck'}:
        return send_from_directory(SAMPLES_FOLDER, f'{sample_name}.jpg')
    return jsonify({'error': f'Sample image for {class_name} not found'}), 404

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'classes': CLASSES,
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # Eagerly load model on startup if present
    try:
        get_model()
        print("Trained model successfully preloaded into memory.")
    except Exception as e:
        print(f"Model will be loaded on demand: {e}")
    app.run(host='0.0.0.0', port=5000, debug=True)
