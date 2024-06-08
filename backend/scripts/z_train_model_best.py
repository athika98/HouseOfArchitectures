# scripts/train_model.py
import sys
import os

# Füge das übergeordnete Verzeichnis zum Python-Pfad hinzu
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from scripts.data_preparation_best import train_generator, validation_generator

# Modell erstellen
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# Modell kompilieren
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Modell trainieren
print("Starte Training...")
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Absoluter Pfad zum Speichern des Modells
model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models/architecture_style_model_10.keras')
print(f"Speichern des Modells unter: {model_save_path}")
# Modell speichern im neuen Keras-Format
model.save(model_save_path)

# Überprüfen, ob das Modell gespeichert wurde
if os.path.exists(model_save_path):
    print(f"Modell erfolgreich gespeichert: {model_save_path}")
else:
    print(f"Fehler beim Speichern des Modells: {model_save_path}")
