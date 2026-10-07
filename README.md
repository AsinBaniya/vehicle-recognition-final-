# NeuroDrive AI — Vehicle Recognition System
## Final Project & Interactive Web Demonstration (Week 8)

An end-to-end deep learning vehicle recognition web application capable of identifying and classifying road vehicles into four target categories:
- **Bus** (Public Transit & Commercial Transport)
- **Car** (Passenger Sedans, SUVs, Coupes)
- **Motorcycle** (Two-Wheeled Motor Vehicles)
- **Truck** (Commercial Freight & Cargo Transport)

---

## 1. Project Overview
The system utilizes a **Deep Convolutional Neural Network (MobileNetV2 Transfer Learning Backbone)** trained on an authentic 400-image vehicle dataset. 
The web interface provides an interactive, client-side dashboard with live drag-and-drop image uploads, confidence score gauges, softmax probability distribution bars across all four classes, and preset test vehicle samples for instant demonstration.

### Key Metrics of the Trained Model (Evaluated on Isolated Test Set)
| Metric | Performance Score |
| :--- | :--- |
| **Test Accuracy** | **86.67%** (52/60 correct) |
| **Macro Precision** | **89.52%** |
| **Macro Recall** | **86.67%** |
| **Macro F1-Score** | **86.25%** |
| **Macro ROC-AUC (One-vs-Rest)** | **0.9863** |
| **Passenger Car Recall** | **100.0%** (15/15) |
| **Bus / Motorcycle Recall** | **93.33%** (14/15) |

---

## 2. Directory Structure
```text
Week_8/
│
├── app.py                      # Flask REST API backend server
├── README.md                   # Complete documentation and user guide
├── requirements.txt            # Runtime dependencies
├── .python-version             # Vercel Python runtime version
│
├── model/                      # Persisted trained model directory
│   ├── best_vehicle_model.h5   # Legacy HDF5 model format
│   └── best_vehicle_model.keras# Native modern Keras model format
│
├── templates/
│   └── index.html              # Modern, responsive HTML5 UI
│
└── public/
    ├── style.css               # Glassmorphic dark-theme CSS3 styles
    ├── script.js               # Client-side upload, fetch, and animation logic
    └── samples/                # Bundled vehicle images for preset buttons
```

---

## 3. Installation & Dependencies

### Prerequisites
- Python 3.9+ (tested and verified on Python 3.11)
- Windows / macOS / Linux

### Install Required Python Libraries
Open a terminal in the project directory and install the required dependencies:
```bash
pip install flask tensorflow pillow numpy scikit-learn
```

---

## 4. How to Run the Application

1. Navigate to the `Week_8` directory:
   ```bash
   cd "Week_8"
   ```

2. Start the Flask application:
   ```bash
   python app.py
   ```

3. Open your web browser and navigate to:
   ```text
   http://127.0.0.1:5000/
   ```
   or
   ```text
   http://localhost:5000/
   ```

---

## 5. How to Upload and Classify an Image

The application supports three intuitive methods for testing images:
1. **Drag and Drop**: Drag any vehicle image (`.jpg`, `.png`, `.jpeg`, `.webp`) directly into the glowing dashed dropzone box.
2. **File Browser**: Click anywhere within the dropzone to open your local file dialog and choose an image.
3. **One-Click Presets**: Click any of the four preset buttons (**Bus**, **Car**, **Motorcycle**, or **Truck**) located right below the dropzone to immediately load a genuine test sample from the dataset and classify it automatically.

Once an image is selected:
- Click the **"Classify Vehicle"** button.
- An animated scanner radar beam activates while feedforward inference runs through the convolutional layers.
- The top predicted class, confidence badge, and four color-coded probability distribution progress bars are rendered dynamically.

---

## 6. How the Prediction Works

When an image is submitted:
1. **Request Intake**: Flask receives the multipart image file via the `/predict` POST route.
2. **Color Mode Normalization**: PIL inspects the image. Any Alpha transparency (RGBA) or indexed palettes (`P`) are safely converted to standard 3-channel RGB.
3. **Lanczos Resizing**: The image is rescaled to exactly $128 \times 128$ pixels using high-fidelity Lanczos-windowed sinc interpolation.
4. **Intensity Scaling**: Pixel values ($0 - 255$) are normalized to $[0.0, 1.0]$ float32 values:
   $$\mathbf{X}_{norm} = \frac{\mathbf{X}_{raw}}{255.0}$$
5. **Batch Expansion**: The tensor is shaped to `(1, 128, 128, 3)` and passed into `model.predict()`.
6. **Softmax Output**: The network computes probability logits across the 4 classes:
   $$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^4 e^{z_j}}$$
7. **JSON Response**: The class with the maximum probability ($\text{argmax}$) and per-class percentages are returned to the client and rendered smoothly.

Uploaded images are processed in memory and are not saved on the server. The result includes a resized JPEG preview as a data URL, so it works with Vercel's temporary function filesystem.

---

## 7. Example Output

### API JSON Response (`/predict`)
```json
{
  "success": true,
  "prediction": "Car",
  "confidence": 98.42,
  "probabilities": {
    "Bus": 0.45,
    "Car": 98.42,
    "Motorcycle": 0.12,
    "Truck": 1.01
  },
  "image_url": "data:image/jpeg;base64,...",
  "meta": {
    "type": "Passenger Vehicle",
    "badge": "Private Transport",
    "icon": "fa-car-side"
  }
}
```

### Visual UI Representation
```text
+-----------------------------------------------------------------------+
|  [🚗] Detected Category: PASSENGER TRANSPORT VEHICLE                  |
|  Class: CAR                                  Confidence: 98.42%      |
+-----------------------------------------------------------------------+
|  Class Probability Distribution:                                      |
|  [🚗] Car        █████████████████████████████████████ 98.42% [TOP]   |
|  [🚚] Truck      █                                      1.01%         |
|  [🚌] Bus                                               0.45%         |
|  [🏍️] Motorcycle                                        0.12%         |
+-----------------------------------------------------------------------+
```

---

## 8. Academic Authorship & Integrity
- **Course**: Machine Learning / Deep Learning
- **Project Structure**: Week 6 (Classification Notebook) | Week 7 (Academic Report) | Week 8 (Web Demo)
- **Dataset**: Balanced 4-Class Vehicle Imagery (`Bus/`, `Car/`, `motorcycle/`, `Truck/`)
- **Framework**: TensorFlow / Keras 2.x, Python Flask, HTML5/CSS3/JavaScript
