# scripts/train_model.py
# Hier 1.9.2	Configure Model’s losses & evaluation metrics angepasst. Folgendes hinzugefügt: Abschnitt Modell kompilieren


import sys
import os

# Suppress warning messages
import warnings
warnings.filterwarnings('ignore')

# Suppress Tensorflow warnings and errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add the parent directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from scripts.data_preparation import train_generator, validation_generator

# Überprüfen der Anzahl der Klassen
print(f"Anzahl der Klassen: {train_generator.num_classes}")

# Load pre-trained ResNet50 model without the top layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Eigenen Schichtenstapel hinzufügen
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)  # Softmax für Mehrklassenklassifikation

# Modell erstellen
model = Model(inputs=base_model.input, outputs=predictions)

# Basis-Modellschichten einfrieren
for layer in base_model.layers:
    layer.trainable = False

# Modell kompilieren
model.compile(
    loss='categorical_crossentropy',  # Verlustfunktion für Mehrklassenklassifikation
    optimizer='adam',                 # Adam-Optimierer
    metrics=['accuracy']              # Evaluationsmetrik: Genauigkeit
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler = LearningRateScheduler(scheduler)

# Modell trainieren
print("Starte Training...")
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr, lr_scheduler]
)

# Absoluter Pfad zum Speichern des Modells
model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models/architecture_style_model_50_0001.keras')
print(f"Speichern des Modells unter: {model_save_path}")
# Modell speichern im neuen Keras-Format
model.save(model_save_path)

# Überprüfen, ob das Modell gespeichert wurde
if os.path.exists(model_save_path):
    print(f"Modell erfolgreich gespeichert: {model_save_path}")
else:
    print(f"Fehler beim Speichern des Modells: {model_save_path}")
