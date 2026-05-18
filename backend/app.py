import os
import string
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import model_from_json
import traceback

app = Flask(__name__)
CORS(app)

# Load model globally
model = None
character = string.ascii_lowercase + "0123456789"
nchar = len(character)

def load_captcha_model():
    global model
    try:
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        json_path = os.path.join(model_dir, 'model.json')
        weights_path = os.path.join(model_dir, 'model_weights.h5')
        
        with open(json_path, 'r') as json_file:
            loaded_model_json = json_file.read()
            
        model = model_from_json(loaded_model_json)
        model.load_weights(weights_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

load_captcha_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
            
        file = request.files['image']
        
        # Read the image file using OpenCV
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400
            
        # Preprocess the image
        img = cv2.resize(img, (200, 50))
        img = img / 255.0
        
        # Make prediction
        res = np.array(model.predict(img[np.newaxis, :, :, np.newaxis]))
        result = np.reshape(res, (5, 36))
        
        k_ind = []
        for i in result:
            k_ind.append(np.argmax(i))
            
        capt = ''
        for k in k_ind:
            capt += character[k]
            
        return jsonify({
            'success': True,
            'prediction': capt
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Captcha API is running"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
