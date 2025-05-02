from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__, static_folder='static')

# Load pre-trained model
model = MobileNetV2(weights='imagenet')

# Serve the frontend
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# Identify image
@app.route('/identify', methods=['POST'])
def identify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = Image.open(io.BytesIO(file.read()))
        img = img.convert('RGB')  # Ensure it's RGB
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array)
        decoded = decode_predictions(predictions, top=1)[0][0]
        label = decoded[1]
        confidence = decoded[2] * 100

        return jsonify({'prediction': f"{label} ({confidence:.2f}%)"})
    
    except Exception as e:
        return jsonify({'error': f"Error processing image: {str(e)}"}), 500

# Optional: serve other static files like images, CSS, JS
@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    app.run(debug=True)
