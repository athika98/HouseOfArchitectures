from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
CORS(app)

MODEL_PATHS = {
    'custom_cnn_model': 'models/custom_cnn_model_10epochs.keras',
    'archinet': 'models/archinet_model_10epochs.keras',
    'resnet': 'models/resnet50_model_10epochs.keras'
}

models = {}
for name, path in MODEL_PATHS.items():
    try:
        print(f"Loading model {name} from path: {path}")
        models[name] = tf.keras.models.load_model(path)
    except Exception as e:
        print(f"Error loading model {name}: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or 'model' not in request.form:
        return jsonify({"error": "No file or model specified"}), 400
    
    file = request.files['file']
    model_name = request.form['model']

    if model_name not in models:
        return jsonify({"error": f"Model {model_name} not found"}), 400

    model = models[model_name]

    # Ensure the /tmp directory exists
    os.makedirs('/tmp', exist_ok=True)
    
    img_path = os.path.join('/tmp', file.filename)
    file.save(img_path)

    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
    except Exception as e:
        return jsonify({"error": f"Error processing image: {e}"}), 500

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    categories = [
        'Achaemenid architecture', 'American craftsman style', 'American Foursquare architecture',
        'Ancient Egyptian architecture', 'Art Deco architecture', 'Art Nouveau architecture',
        'Baroque architecture', 'Bauhaus architecture', 'Beaux-Arts architecture',
        'Byzantine architecture', 'Chicago school architecture', 'Colonial architecture',
        'Deconstructivism', 'Edwardian architecture', 'Georgian architecture',
        'Gothic architecture', 'Greek Revival architecture', 'International style',
        'Novelty architecture', 'Palladian architecture', 'Postmodern architecture',
        'Queen Anne architecture', 'Romanesque architecture', 'Russian Revival architecture',
        'Tudor Revival architecture'
    ]

    predicted_label = categories[predicted_class[0]]
    return jsonify({"predictions": predicted_label})

@app.route('/results/<path:path>')
def send_results(path):
    return send_from_directory('results', path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
