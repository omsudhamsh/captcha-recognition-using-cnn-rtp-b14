import os
import string
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow import keras
from tensorflow.keras import layers, Model
import traceback

app = Flask(__name__)
CORS(app)

# Load model globally
model = None
character = string.ascii_lowercase + "0123456789"
nchar = len(character)
IMG_HEIGHT, IMG_WIDTH = 50, 200

def create_model():
    """Create the original CAPTCHA recognition model architecture"""
    img = layers.Input(shape=(50, 200, 1))
    conv1 = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(img)
    mp1 = layers.MaxPooling2D(padding='same')(conv1)
    conv2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp1)
    mp2 = layers.MaxPooling2D(padding='same')(conv2)
    conv3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp2)
    bn = layers.BatchNormalization()(conv3)
    mp3 = layers.MaxPooling2D(padding='same')(bn)
    flat = layers.Flatten()(mp3)
    
    outs = []
    for _ in range(5):
        dens1 = layers.Dense(64, activation='relu')(flat)
        drop = layers.Dropout(0.5)(dens1)
        res = layers.Dense(nchar, activation='sigmoid')(drop)
        outs.append(res)
    
    model = Model(img, outs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model

def load_captcha_model():
    global model
    try:
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        # Try to load improved model first (if trained)
        improved_path = os.path.join(model_dir, 'improved_model.h5')
        if os.path.exists(improved_path):
            print("Loading improved model (transfer learning)...")
            model = keras.models.load_model(improved_path)
            print("✓ Improved model loaded successfully!")
            return
        
        # Fallback to original model
        weights_path = os.path.join(model_dir, 'model_weights.h5')
        model = create_model()
        
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            print("✓ Original model loaded successfully.")
        else:
            print(f"Warning: Model weights not found at {weights_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()

def preprocess_image(img):
    """Enhanced preprocessing to handle various CAPTCHA formats"""
    try:
        # Handle different image formats
        if img is None:
            return None
        
        # If color image, convert to grayscale
        if len(img.shape) == 3:
            if img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            elif img.shape[2] == 3:  # RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Normalize image size (adaptive)
        h, w = img.shape
        aspect_ratio = w / h
        target_aspect = IMG_WIDTH / IMG_HEIGHT
        
        if aspect_ratio > target_aspect:
            # Width dominant - resize width to target
            new_w = IMG_WIDTH
            new_h = int(IMG_WIDTH / aspect_ratio)
            img = cv2.resize(img, (new_w, new_h))
            # Pad height
            pad = (IMG_HEIGHT - new_h) // 2
            img = cv2.copyMakeBorder(img, pad, IMG_HEIGHT - new_h - pad, 0, 0, 
                                     cv2.BORDER_CONSTANT, value=255)
        else:
            # Height dominant - resize height to target
            new_h = IMG_HEIGHT
            new_w = int(IMG_HEIGHT * aspect_ratio)
            img = cv2.resize(img, (new_w, new_h))
            # Pad width
            pad = (IMG_WIDTH - new_w) // 2
            img = cv2.copyMakeBorder(img, 0, 0, pad, IMG_WIDTH - new_w - pad,
                                     cv2.BORDER_CONSTANT, value=255)
        
        # Final resize to exact dimensions
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        
        # Normalize to 0-1
        img = img.astype('float32') / 255.0
        
        return img
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None

load_captcha_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Please restart the server.'}), 500
            
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
            
        file = request.files['image']
        
        # Read the image file using OpenCV
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Preprocess image
        img = preprocess_image(img)
        if img is None:
            return jsonify({'error': 'Failed to preprocess image'}), 400
        
        # Add batch and channel dimensions
        img_input = img[np.newaxis, :, :, np.newaxis]
        
        # Make prediction
        res = np.array(model.predict(img_input, verbose=0))
        result = np.reshape(res, (5, nchar))
        
        # Get character indices
        k_ind = []
        confidences = []
        for i in result:
            idx = np.argmax(i)
            k_ind.append(idx)
            confidences.append(float(i[idx]))
        
        # Build CAPTCHA string
        capt = ''
        for k in k_ind:
            capt += character[k]
        
        # Calculate overall confidence
        avg_confidence = np.mean(confidences)
        
        return jsonify({
            'success': True,
            'prediction': capt,
            'confidence': float(avg_confidence),
            'character_confidences': confidences
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "message": "Captcha API is running",
        "model": "improved" if "improved" in str(type(model)) else "original"
    })

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        "app": "CAPTCHA Recognition API",
        "version": "2.0-upgraded",
        "features": [
            "Transfer learning (MobileNetV2)",
            "Data augmentation support",
            "Multiple CAPTCHA format support",
            "Adaptive preprocessing",
            "Confidence scores"
        ],
        "supported_formats": "PNG, JPEG, GIF, BMP (grayscale or color)",
        "characters": character
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
