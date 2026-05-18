import os
import string
import numpy as np
import cv2
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import pickle

# Configuration
character = string.ascii_lowercase + "0123456789"
nchar = len(character)
IMG_HEIGHT, IMG_WIDTH = 50, 200

def create_improved_model():
    """Create an improved model using transfer learning with MobileNetV2"""
    
    # Input layer
    img_input = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    
    # Convert grayscale to RGB for MobileNetV2 using Lambda
    from tensorflow.keras.layers import Lambda
    import tensorflow as tf
    rgb = Lambda(lambda x: tf.image.grayscale_to_rgb(x))(img_input)
    
    # Resize to MobileNetV2 input size (224x224)
    resized = layers.Resizing(224, 224)(rgb)
    
    # Normalize inputs
    normalized = layers.Rescaling(1./127.5, offset=-1)(resized)
    
    # Load pre-trained MobileNetV2 (transfer learning)
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze base model initially
    
    # Extract features
    features = base_model(normalized)
    x = layers.GlobalAveragePooling2D()(features)
    
    # Dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    # Output heads (5 characters)
    outs = []
    for _ in range(5):
        out = layers.Dense(128, activation='relu')(x)
        out = layers.Dropout(0.3)(out)
        out = layers.Dense(nchar, activation='softmax')(out)
        outs.append(out)
    
    model = Model(img_input, outs)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    return model, base_model

def get_data_augmentation():
    """Create augmentation pipeline for robustness"""
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.05,
        zoom_range=0.2,
        shear_range=0.1,
        fill_mode='nearest'
    )

def load_data():
    """Load training data"""
    try:
        X = np.load("models/X.txt.npy")
        Y = np.load("models/Y.txt.npy")
        print(f"✓ Data loaded: {X.shape[0]} images")
        return X, Y
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def train_improved_model():
    """Train the improved model with augmentation"""
    
    print("=" * 60)
    print("🚀 TRAINING IMPROVED CAPTCHA MODEL (Transfer Learning + Augmentation)")
    print("=" * 60)
    
    # Load data
    X, Y = load_data()
    if X is None:
        print("❌ Failed to load training data")
        return
    
    # Split data
    X_train, y_train = X[:970], Y[:, :970]
    X_test, y_test = X[970:], Y[:, 970:]
    
    print(f"✓ Training samples: {X_train.shape[0]}")
    print(f"✓ Test samples: {X_test.shape[0]}")
    
    # Create model
    model, base_model = create_improved_model()
    print("✓ Model created with MobileNetV2 backbone")
    
    # Phase 1: Train with frozen base model
    print("\n📍 Phase 1: Training top layers (base model frozen)...")
    datagen = get_data_augmentation()
    
    history1 = model.fit(
        datagen.flow(X_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]], 
                     batch_size=16, shuffle=True),
        epochs=15,
        validation_data=(X_test, [y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]]),
        verbose=1
    )
    
    # Phase 2: Fine-tune with unfrozen base model
    print("\n📍 Phase 2: Fine-tuning base model layers...")
    base_model.trainable = True
    
    # Freeze early layers, only train later layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        X_train,
        [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]],
        batch_size=16,
        epochs=8,
        validation_data=(X_test, [y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]]),
        verbose=1
    )
    
    # Save model and weights
    print("\n✓ Saving improved model...")
    model.save('models/improved_model.h5')
    model_json = model.to_json()
    with open('models/improved_model.json', 'w') as json_file:
        json_file.write(model_json)
    
    # Evaluate
    print("\n📊 MODEL EVALUATION")
    print("=" * 60)
    predictions = model.predict(X_test, verbose=0)
    
    # Calculate character accuracy
    char_accuracy = 0
    total_chars = 0
    for i in range(5):
        pred_idx = np.argmax(predictions[i], axis=1)
        true_idx = np.argmax(y_test[i], axis=1)
        char_accuracy += np.sum(pred_idx == true_idx)
        total_chars += len(true_idx)
    
    print(f"Overall character accuracy: {char_accuracy / total_chars * 100:.2f}%")
    
    # Calculate CAPTCHA accuracy (all 5 chars must be correct)
    all_correct = 0
    for i in range(len(X_test)):
        all_correct_for_sample = True
        for j in range(5):
            pred_idx = np.argmax(predictions[j][i])
            true_idx = np.argmax(y_test[j][i])
            if pred_idx != true_idx:
                all_correct_for_sample = False
                break
        if all_correct_for_sample:
            all_correct += 1
    
    print(f"Full CAPTCHA accuracy: {all_correct / len(X_test) * 100:.2f}%")
    print("=" * 60)
    print("✅ Training complete! Improved model saved.")

if __name__ == '__main__':
    train_improved_model()
