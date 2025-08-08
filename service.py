from flask import Flask, request, jsonify
from fastai.vision.all import *
from pathlib import Path
import tempfile

app = Flask(__name__)

# Load the trained model
model_path = Path(__file__).parent.parent.parent / 'assets' / 'birds' / 'bird_classifier.pkl'
learn = load_learner(model_path)

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
    app.run(debug=True)