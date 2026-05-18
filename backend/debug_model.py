import os
import tensorflow as tf
from tensorflow.keras.models import model_from_json

model_dir = os.path.join(os.path.dirname(__file__), 'models')
json_path = os.path.join(model_dir, 'model.json')
weights_path = os.path.join(model_dir, 'model_weights.h5')

try:
    with open(json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
        
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
