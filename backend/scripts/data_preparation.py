# scripts/data_preparation.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter
import os

# Absolute path to the data directory
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')

# Image dimensions
img_height, img_width = 224, 224
batch_size = 32

# Create data generators
## Old version - without data augmentation
'''
# Create data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
'''

# New version - with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize the image data to the range [0, 1]
    shear_range=0.2,  # Apply shear transformations
    zoom_range=0.2,  # Apply zoom transformations
    horizontal_flip=True,  # Allow horizontal flipping of images
    rotation_range=20,  # Rotate images up to 20 degrees
    width_shift_range=0.2,  # Shift images horizontally by up to 20% of the width
    height_shift_range=0.2,  # Shift images vertically by up to 20% of the height
    brightness_range=[0.8, 1.2],  # Adjust brightness randomly within this range
    validation_split=0.2  # Use 20% of the data for validation
)

# Generate batches of tensor image data for training
train_generator = train_datagen.flow_from_directory(
    data_dir,  # Path to the data directory
    target_size=(img_height, img_width),  # Resize all images to 224x224
    batch_size=batch_size,  # Number of images to return in each batch
    class_mode='categorical',  # Return one-hot encoded labels
    subset='training'  # Use this subset of the data for training
)

# Generate batches of tensor image data for validation
validation_generator = train_datagen.flow_from_directory(
    data_dir,  # Path to the data directory
    target_size=(img_height, img_width),  # Resize all images to 224x224
    batch_size=batch_size,  # Number of images to return in each batch
    class_mode='categorical',  # Return one-hot encoded labels
    subset='validation'  # Use this subset of the data for validation
)

# Calculate Class weights (because not every folder has the same amount of pictures)

def calculate_class_weights(generator):
    counter = Counter(generator.classes)
    total_samples = sum(counter.values())
    class_weight = {cls: total_samples / (len(counter) * count) for cls, count in counter.items()}
    return class_weight

# Calculate class weights
class_weight = calculate_class_weights(train_generator)
print("Class weights:", class_weight)
