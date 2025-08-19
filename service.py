from flask import Flask, request, jsonify, Response, stream_with_context
from fastai.vision.all import *
from pathlib import Path
import tempfile
from huggingface_hub import from_pretrained_fastai
import requests
import json


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



@app.route('/ollama', methods=['GET'])
def ollama_generate():
    """Proxy a prompt to local Ollama and return the response.

    Query params:
      - prompt (required): The text prompt to send to the model
      - model (optional): Ollama model name; defaults to env OLLAMA_MODEL or 'llama3.1'
    """
    prompt = request.args.get('prompt', type=str)
    if not prompt:
        return jsonify({'error': "Missing required query parameter 'prompt'"}), 400

    model = request.args.get('model') or os.environ.get('OLLAMA_MODEL', 'llama3.2')
    ollama_url = os.environ.get('OLLAMA_URL', 'http://localhost:11434')

    try:
        resp = requests.post(
            f"{ollama_url.rstrip('/')}/api/generate",
            json={
                'model': model,
                'prompt': prompt,
                'stream': False
            },
            timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        return jsonify({
            'model': model,
            'response': data.get('response'),
            'info': {k: data.get(k) for k in ('created_at','total_duration','load_duration','eval_count','eval_duration') if k in data}
        })
    except requests.exceptions.RequestException as e:
        return jsonify({'error': str(e)}), 502


@app.route('/ollama/stream', methods=['GET'])
def ollama_stream():
    """Stream tokens from local Ollama via Server-Sent Events (SSE).

    Query params:
      - prompt (required): The text prompt to send to the model
      - model (optional): Ollama model name; defaults to env OLLAMA_MODEL or 'llama3.1'
    """
    prompt = request.args.get('prompt', type=str)
    if not prompt:
        return jsonify({'error': "Missing required query parameter 'prompt'"}), 400

    model = request.args.get('model') or os.environ.get('OLLAMA_MODEL', 'llama3.2')
    ollama_url = os.environ.get('OLLAMA_URL', 'http://localhost:11434')

    def event_stream():
        try:
            with requests.post(
                f"{ollama_url.rstrip('/')}/api/generate",
                json={'model': model, 'prompt': prompt, 'stream': True},
                stream=True,
                timeout=300,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    # Stream token chunks as JSON object, with encoded newlines and explicit event name
                    if 'response' in data and data['response']:
                        chunk = str(data['response'])
                        yield f"event: update\ndata: {json.dumps({'content': chunk})}\n\n"
                    # When done, send a final event with basic stats
                    if data.get('done'):
                        info = {k: data.get(k) for k in (
                            'created_at','total_duration','load_duration','eval_count','eval_duration'
                        ) if k in data}
                        yield f"event: done\ndata: {json.dumps(info)}\n\n"
                        break
        except requests.exceptions.RequestException as e:
            yield f"event: error\ndata: {str(e)}\n\n"

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return Response(stream_with_context(event_stream()), mimetype='text/event-stream', headers=headers)


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
