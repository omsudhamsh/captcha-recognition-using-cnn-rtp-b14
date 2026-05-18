import os
import string
import numpy as np
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

character = string.ascii_lowercase + '0123456789'
nchar = len(character)

# Load data
print('Loading data...')
X = np.load('models/X.txt.npy')
Y = np.load('models/Y.txt.npy')
X_train, y_train = X[:970], Y[:, :970]
X_test, y_test = X[970:], Y[:, 970:]

print('=' * 60)
print('🚀 TRAINING IMPROVED CAPTCHA MODEL (Transfer Learning)')
print('=' * 60)
print(f'Training samples: {X_train.shape[0]}')
print(f'Test samples: {X_test.shape[0]}')

# Build model
print('Building model with MobileNetV2 backbone...')
img_input = layers.Input(shape=(50, 200, 1))
rgb = layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x))(img_input)
resized = layers.Resizing(224, 224)(rgb)
normalized = layers.Rescaling(1./127.5, offset=-1)(resized)
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False
features = base_model(normalized)
x = layers.GlobalAveragePooling2D()(features)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
outs = [layers.Dense(nchar, activation='softmax')(layers.Dropout(0.3)(layers.Dense(128, activation='relu')(x))) for _ in range(5)]
model = Model(img_input, outs)
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))

print('✓ Model created')
print('\n📍 Phase 1: Training top layers (base frozen)...')
model.fit(X_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]], batch_size=16, epochs=12, validation_data=(X_test, [y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]]), verbose=1)

print('\n📍 Phase 2: Fine-tuning (unfreezing base model)...')
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001))
model.fit(X_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]], batch_size=16, epochs=8, validation_data=(X_test, [y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]]), verbose=1)

print('\n✓ Saving improved model...')
model.save('models/improved_model.h5')

predictions = model.predict(X_test, verbose=0)
char_accuracy = sum(np.sum(np.argmax(predictions[i], axis=1) == np.argmax(y_test[i], axis=1)) for i in range(5)) / (5 * len(X_test)) * 100
full_correct = sum(1 for idx in range(len(X_test)) if all(np.argmax(predictions[j][idx]) == np.argmax(y_test[j][idx]) for j in range(5)))
full_accuracy = full_correct / len(X_test) * 100

print(f'\n📊 Character accuracy: {char_accuracy:.2f}%')
print(f'📊 Full CAPTCHA accuracy: {full_accuracy:.2f}%')
print('=' * 60)
print('✅ UPGRADE COMPLETE!')
