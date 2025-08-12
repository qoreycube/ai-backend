from flask import Flask, request, jsonify
from fastai.vision.all import *
from pathlib import Path
import tempfile
from huggingface_hub import from_pretrained_fastai


import os

app = Flask(__name__)


# Load the trained model
model_dir = Path(__file__).parent
model_files = list(model_dir.glob('*.pkl'))

# Load fastai model as before
if not model_files:
    raise FileNotFoundError("No .pkl model file found in the directory.")
model_path = model_files[0]
learn = load_learner(model_path)
learn_hf = from_pretrained_fastai("edwinhung/bird_classifier")

# Load HuggingFace bird classifier pipeline
@app.route('/hf_predict', methods=['POST'])
def hf_predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    img_file = request.files['image']

    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img_path = Path(tmp.name)
            img_file.save(img_path)


        # Make prediction
        pred_class, pred_idx, probs = learn_hf.predict(img_path)

        # Get top 3 predictions
        top3_idx = probs.argsort(descending=True)[:3]
        top3 = []
        for idx in top3_idx:
            species_name = learn_hf.dls.vocab[idx]
            confidence = float(probs[idx])
            top3.append({'species': species_name, 'confidence': confidence})

        return jsonify({
            'top3': top3,
            'predicted_species': str(pred_class),
            'confidence': float(probs[pred_idx])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    

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

        # Get top 3 predictions
        top3_idx = probs.argsort(descending=True)[:3]
        top3 = []
        for idx in top3_idx:
            species_name = learn.dls.vocab[idx]
            confidence = float(probs[idx])
            top3.append({'species': species_name, 'confidence': confidence})

        return jsonify({
            'top3': top3,
            'predicted_species': str(pred_class),
            'confidence': float(probs[pred_idx])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 9000))
    app.run(debug=True, host='0.0.0.0', port=port)
