# scripts/train_resnet_model.py
import sys
import os
from datetime import datetime
import json

# Suppress Tensorflow warnings and errors
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from data_preparation import train_generator, validation_generator


# Absolute path to the data directory
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Check the number of classes
print(f"Number of classes: {train_generator.num_classes}")

# Load pre-trained ResNet50 model without the top layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001) # Lernrate manuell angepasst. Old: optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler = LearningRateScheduler(scheduler)

# Train the model
print("Starte Training...")
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr]
)

# Absolute path to save the model
model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models/resnet50_model_20epochs.keras')
print(f"Speichern des Modells unter: {model_save_path}")
model.save(model_save_path)

# Get the current date and time
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Save the results
results = {
    'model': 'ResNet50',
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