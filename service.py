from flask import Flask, request, jsonify
from fastai.vision.all import *
from pathlib import Path
import tempfile
import os

app = Flask(__name__)


# Load the trained model
model_path = Path(__file__).parent / 'bird_classifier_3_species.pkl'
learn = load_learner(model_path)

# Helper to load species from bird_species.txt
def load_species():
    species_path = Path(__file__).parent / 'bird_species.txt'
    if not species_path.exists():
        return []
    with open(species_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]
    
@app.route('/species', methods=['GET'])
def get_species():
    species = load_species()
    return jsonify({'species': species})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    img_file = request.files['image']

    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img_path = Path(tmp.name)
            img_file.save(img_path)

        # Make prediction
        pred_class, pred_idx, probs = learn.predict(img_path)

        return jsonify({
            'predicted_species': str(pred_class),
            'confidence': float(probs[pred_idx])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)