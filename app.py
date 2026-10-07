import os
import json
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)

# Constants
CLASSES = ['Bus', 'Car', 'Motorcycle', 'Truck']
TARGET_SIZE = (128, 128)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

# Load the trained model
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
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
    Safely preprocess uploaded image:
    Handles RGB, RGBA, Grayscale, and Palette modes.
    Resizes using high quality Lanczos interpolation to 128x128.
    Normalizes pixel values to [0.0, 1.0].
    """
    img = Image.open(image_file)
    if img.mode != 'RGB':
        img = img.convert('RGBA').convert('RGB')
    img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # Shape (1, 128, 128, 3)
    return arr

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided in request.'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected.'}), 400
    
    try:
        # Save temporary uploaded image for display
        filename = f"upload_{os.urandom(6).hex()}_{file.filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.seek(0)
        file.save(save_path)
        
        # Preprocess and predict
        processed_input = preprocess_image(save_path)
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
            'image_url': f"/static/uploads/{filename}",
            'meta': vehicle_meta.get(pred_class, {})
        })
        
    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

@app.route('/api/sample/<class_name>')
def get_sample(class_name):
    # Map class name properly
    cls_map = {'bus': 'Bus', 'car': 'Car', 'motorcycle': 'motorcycle', 'truck': 'Truck'}
    target_cls = cls_map.get(class_name.lower(), class_name)
    
    # Check dataset directory
    dataset_dirs = [
        r"C:\Users\admin\OneDrive\Desktop\week2\Dataset",
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset'))
    ]
    for d in dataset_dirs:
        folder = os.path.join(d, target_cls)
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if files:
                chosen = files[0]
                return send_from_directory(folder, chosen)
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
