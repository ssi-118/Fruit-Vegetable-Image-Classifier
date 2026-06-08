import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pathlib import Path
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image

# Limit TensorFlow CPU thread usage to reduce memory pressure in low-memory environments
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

app = Flask(__name__)

# --- Configuration ---
# Ensure storage folders exist
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / 'static' / 'uploads'
MODEL_PATH = BASE_DIR / 'model' / 'fruit_model.h5'
LABELS_PATH = BASE_DIR / 'model' / 'labels.txt'

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}

# --- Load Model & Labels ---
# We load these globally so they stay in memory for fast predictions
model = None
labels = []
model_load_error = None

try:
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines()]
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model and Labels loaded successfully.")
except Exception as e:
    model_load_error = str(e)
    print(f"Error loading model: {model_load_error}")
    print("Ensure you have run train_model.py successfully first.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or not labels:
        return (
            "Model is not loaded. "
            f"Expected model at: {MODEL_PATH}. "
            f"Expected labels at: {LABELS_PATH}. "
            f"Load error: {model_load_error}"
        ), 500

    # 1. Check if a file was uploaded
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # 2. Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            app.logger.info(f"Processing uploaded image: {filename}")
            # 3. Preprocess the image
            # We use 224x224 to match MobileNetV2 input size
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)

            # MobileNetV2 expects input in range [-1, 1]
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            # 4. Make Prediction
            predictions = model.predict(img_array, verbose=0)[0]

            # 5. Get Top-3 Results
            top_indices = predictions.argsort()[-3:][::-1]
            results = []
            for i in top_indices:
                results.append({
                    "class": labels[i],
                    "conf": round(float(predictions[i]) * 100, 2)
                })

            app.logger.info("Prediction completed successfully.")

            # 6. Render Result Page
            # We pass 'img_filename' which the template uses with url_for
            return render_template('result.html',
                                   results=results,
                                   img_filename=filename)

        except Exception as e:
            app.logger.exception("Error during processing")
            return f"Error during processing: {str(e)}"
    else:
        return "Invalid file type. Please upload a JPG, PNG, or WebP image."

if __name__ == '__main__':
    # Threaded=False can sometimes prevent issues with TensorFlow on some Windows machines
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', debug=False, port=port)
