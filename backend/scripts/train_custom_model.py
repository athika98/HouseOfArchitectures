# scripts/train_custom_model.py
import os
from datetime import datetime
import json

# Suppress Tensorflow warnings and errors
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from data_preparation import train_generator, validation_generator

# Check the number of classes
print(f"Number of classes: {train_generator.num_classes}")

# Create the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator
)

# Absolute path to save the model
model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models/custom_cnn_model_20epochs.keras')
print(f"Saving the model to: {model_save_path}")
model.save(model_save_path)

# Check if the model was saved successfully
if os.path.exists(model_save_path):
    print(f"Model successfully saved: {model_save_path}")
else:
    print(f"Error saving the model: {model_save_path}")

# Get the current date and time
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Save the results
results = {
    'model': 'Custom CNN Model',
    'date_time': current_time,
    'epochs': 20,
    'train_accuracy': history.history['accuracy'],
    'val_accuracy': history.history['val_accuracy'],
    'train_loss': history.history['loss'],
    'val_loss': history.history['val_loss']
}

# Ensure the results directory exists
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../results')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Load existing results if they exist
results_save_path = os.path.join(results_dir, 'results.json')
print(f"Saving results to: {results_save_path}")
if os.path.exists(results_save_path):
    try:
        with open(results_save_path, 'r') as f:
            all_results = json.load(f)
            print(f"Loaded existing results: {all_results}")
    except json.JSONDecodeError as e:
        print(f"Error loading JSON: {e}")
        all_results = []
else:
    all_results = []

# Append new results
all_results.append(results)
print(f"Appending new results: {results}")

# Save all results
with open(results_save_path, 'w') as f:
    json.dump(all_results, f, indent=4)
    print(f"Results saved to {results_save_path}")
