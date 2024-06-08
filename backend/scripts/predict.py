# scripts/predict.py
# Use : python predict.py <model_name> <img_path> (e.g. python predict.py resnet ../data/Gothic architecture/image_Gothic_architecture_01.jpg)

import os
import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from data_preparation import train_generator

# Dictionary with paths to the models
MODEL_PATHS = {
    'custom_cnn_model': '../models/custom_cnn_model_10epochs.keras',
    'archinet': '../models/archinet_model_10epochs.keras',
    'resnet': '../models/resnet50_model_10epochs.keras',
    'inceptionv3': '../models/inceptionv3_model_20epochs.keras'
}

# Function to load the model
def load_model(model_name):
    model_path = MODEL_PATHS.get(model_name)
    if not model_path or not os.path.exists(model_path):
        raise ValueError(f"Model '{model_name}' not found.")
    return tf.keras.models.load_model(model_path)

# Load class names
class_names = list(train_generator.class_indices.keys())

def predict_image(model, img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict.py <model_name> <img_path>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    img_path = sys.argv[2]

    try:
        # Load the model
        model = load_model(model_name)
        # Print the predicted class
        print(predict_image(model, img_path))
    except Exception as e:
        print(e)
        sys.exit(1)
