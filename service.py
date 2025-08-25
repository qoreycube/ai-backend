from flask import Flask, request, jsonify, Response, stream_with_context, send_file
from fastai.vision.all import *
from pathlib import Path
import tempfile
from huggingface_hub import from_pretrained_fastai
import requests
import json
import cv2
import numpy as np
from PIL import Image
import base64
import io


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
    """Return list of bird species from bird_species.txt (if present)."""
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

    model = request.args.get('model') or os.environ.get('OLLAMA_MODEL', 'llama3.1')
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

    model = request.args.get('model') or os.environ.get('OLLAMA_MODEL', 'llama3.1')
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


@app.route('/outline', methods=['POST'])
def outline_image():
    """Extract outline/edges from an image using OpenCV.

    Form-data params:
      - image: file (required)
      - low_threshold: int (default: 50) - lower threshold for Canny
      - high_threshold: int (default: 150) - upper threshold for Canny
      - blur_kernel: int (default: 5) - Gaussian blur kernel size (odd)
      - invert_colors: bool (default: false) - if true, invert output colors
        (also accepts legacy 'invert' which remains supported)
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400

        # Get parameters with defaults
        low_threshold = int(request.form.get('low_threshold', 50))
        high_threshold = int(request.form.get('high_threshold', 150))
        blur_kernel = int(request.form.get('blur_kernel', 5))

        # Support both legacy 'invert' and new 'invert_colors' param
        invert_param = request.form.get('invert', None)
        invert_colors_param = request.form.get('invert_colors', None)

        def _to_bool(v):
            if v is None:
                return False
            return str(v).lower() in ('1', 'true', 'yes', 'y')

        invert = _to_bool(invert_param) or _to_bool(invert_colors_param)

        # Validate parameters
        if blur_kernel % 2 == 0:
            blur_kernel += 1  # Ensure odd number for kernel

        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, low_threshold, high_threshold)

        # Invert if requested (black lines on white background)
        if invert:
            edges = cv2.bitwise_not(edges)

        # Convert back to PIL Image
        pil_image = Image.fromarray(edges)

        # Save to bytes and return
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        return send_file(img_buffer, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return jsonify({
        'message': 'Bird Watcher Backend API',
        'endpoints': {
            '/predict': 'POST - Upload image for bird species prediction',
            '/hf_predict': 'POST - Upload image for HuggingFace model prediction', 
            '/species': 'GET - List all bird species',
            '/ollama': 'POST - Chat with local LLM (JSON: {message: "your question"})',
            '/ollama/stream': 'POST - Stream chat with local LLM (JSON: {message: "your question"})',
            '/outline': 'POST - Extract image outline/edges (form-data: image file + optional params)',
            '/coloring_xdog': 'POST - Deterministic coloring/outline generator (form-data params documented below)',
            '/coloring_hf': 'POST - Local coloring/book generator (form-data params documented below)'
        },
        'outline_params': {
            'low_threshold': 'int (default: 50) - Lower threshold for Canny edge detection',
            'high_threshold': 'int (default: 150) - Upper threshold for Canny edge detection',
            'blur_kernel': 'int (default: 5) - Gaussian blur kernel size (odd numbers only)',
            'invert_colors': 'bool (default: false) - Invert the output colors (black/white)'
        },
        'coloring_xdog_params': {
            'resize_max': 'int (default: 1024) - Max image dimension for processing',
            'bilateral_d': 'int (default: 9) - Bilateral filter diameter',
            'bilateral_sigma_color': 'int (default: 75)',
            'bilateral_sigma_space': 'int (default: 75)',
            'k_clusters': 'int (default: 6) - Color quantization clusters',
            'canny_low': 'int (default: 50)',
            'canny_high': 'int (default: 150)',
            'line_thickness': 'int (default: 2)',
            'min_area': 'int (default: 300)',
            'invert_colors': 'bool (default: false) - Invert final image colors'
        },
        'coloring_hf_params': {
            'k': 'int (default: 6) - Clusters for quantization',
            'line_thickness': 'int (default: 2)',
            'min_area': 'int (default: 300)',
            'invert_colors': 'bool (default: false) - Invert final image colors'
        }
    })


@app.route('/coloring_xdog', methods=['POST'])
def coloring_xdog():
    """New non-XDoG deterministic coloring-book generator.

    This implementation focuses on producing clean, contour-style outlines
    without learned models. It combines bilateral smoothing, gradient magnitude,
    contour extraction from quantized regions, and conservative morphological
    cleanup to produce robust black-line-on-white results.

    Form-data params (all optional except image):
      - image: file (required)
      - resize_max: int (default 1024) - max image dimension for processing
      - bilateral_d: int (default 9) - bilateral filter diameter
      - bilateral_sigma_color: int (default 75)
      - bilateral_sigma_space: int (default 75)
      - k_clusters: int (default 6) - color quantization clusters
      - canny_low: int (default 50)
      - canny_high: int (default 150)
      - line_thickness: int (default 2)
    - min_area: int (default 300)
    - invert_colors: bool (default: false) - Invert final image colors (black/white)
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400

        resize_max = int(request.form.get('resize_max', 1024))
        bilateral_d = int(request.form.get('bilateral_d', 9))
        bilateral_sigma_color = int(request.form.get('bilateral_sigma_color', 75))
        bilateral_sigma_space = int(request.form.get('bilateral_sigma_space', 75))
        k_clusters = int(request.form.get('k_clusters', 6))
        canny_low = int(request.form.get('canny_low', 50))
        canny_high = int(request.form.get('canny_high', 150))
        line_thickness = int(request.form.get('line_thickness', 2))
        min_area = int(request.form.get('min_area', 300))

        data = file.read()
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Resize to limit max dimension
        h, w = img.shape[:2]
        max_dim = max(h, w)
        if max_dim > resize_max:
            scale = resize_max / float(max_dim)
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

        # Strong bilateral filtering to preserve edges while smoothing regions
        smooth = cv2.bilateralFilter(img, d=bilateral_d, sigmaColor=bilateral_sigma_color, sigmaSpace=bilateral_sigma_space)

        # Convert to grayscale and compute gradient magnitude (Sobel)
        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        # Normalize gradient to 0..255
        grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Canny on gradient for crisp edges
        edges_canny = cv2.Canny(grad, canny_low, canny_high)
        edges_canny = cv2.bitwise_not(edges_canny)  # invert so black lines on white

        # Color quantization via K-means to extract region boundaries
        Z = smooth.reshape((-1, 3)).astype(np.float32)
        kq = max(2, min(12, k_clusters))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
        _, labels, centers = cv2.kmeans(Z, kq, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quant = centers[labels.flatten()].reshape(smooth.shape)
        quant_gray = cv2.cvtColor(quant, cv2.COLOR_BGR2GRAY)

        # Extract contours from quantized regions at multiple thresholds
        canvas = np.ones_like(quant_gray) * 255
        kernel = np.ones((3,3), np.uint8)
        for thresh_val in [50, 100, 150, 200]:
            _, th = cv2.threshold(quant_gray, thresh_val, 255, cv2.THRESH_BINARY)
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > min_area:
                    eps = 0.01 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, eps, True)
                    cv2.drawContours(canvas, [approx], -1, 0, line_thickness)

        # Combine contours and Canny edges: prefer strong canny lines but fill in from contours
        combined = cv2.bitwise_and(canvas, edges_canny)

        # Clean and thin lines: morphological opening then slight erosion to thin
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        if line_thickness > 1:
            thin_k = np.ones((line_thickness, line_thickness), np.uint8)
            combined = cv2.dilate(combined, thin_k, iterations=1)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        # Final safety: if mostly white, fallback to a softer Canny of original gray
        mean_val = float(combined.mean())
        if mean_val > 250:
            fb = cv2.Canny(cv2.medianBlur(gray,3), max(10, canny_low//2), max(50, canny_high//2))
            fb = cv2.bitwise_not(fb)
            combined = cv2.dilate(fb, np.ones((max(1,line_thickness),)*2, np.uint8), iterations=1)

        # Optionally invert colors if requested
        invert_colors = request.form.get('invert_colors', None)
        if str(invert_colors).lower() in ('1', 'true', 'yes', 'y'):
            combined = cv2.bitwise_not(combined)

        pil_img = Image.fromarray(combined)
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/coloring_hf', methods=['POST'])
def coloring_hf():
    """Local, model-free coloring-book generator.

    This endpoint no longer calls Hugging Face. It uses local image processing
    to produce a coloring-book style page (mean-shift / quantization + edges).

    Form-data params:
      - image: file (required)
      - k: int (optional, default 6) - number of color clusters for quantization
      - line_thickness: int (optional, default 2)
    - min_area: int (optional, default 300)
    - invert_colors: bool (default: false) - Invert final image colors (black/white)
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400

        k_clusters = int(request.form.get('k', 6))
        line_thickness = int(request.form.get('line_thickness', 2))
        min_area = int(request.form.get('min_area', 300))

        data = file.read()
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Resize for performance
        h, w = img.shape[:2]
        max_dim = max(h, w)
        if max_dim > 1200:
            scale = 1200.0 / max_dim
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

        # Edge-preserving smoothing to preserve important edges while simplifying regions
        shifted = cv2.pyrMeanShiftFiltering(img, sp=21, sr=51)

        # Convert to grayscale and get multi-scale edges
        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)

        # Adaptive threshold captures soft edges; Canny captures strong edges
        edges_adapt = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 9, 2)
        edges_canny = cv2.Canny(blurred, 50, 150)

        # Combine edges and invert so we have black lines on white background
        edges = cv2.bitwise_or(edges_adapt, edges_canny)
        edges = cv2.bitwise_not(edges)

        # Clean edges
        kernel = np.ones((2,2), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Quantize colors with K-means to get clean regions
        Z = shifted.reshape((-1, 3)).astype(np.float32)
        k = max(2, min(12, k_clusters))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quant = centers[labels.flatten()].reshape(shifted.shape)
        quant_gray = cv2.cvtColor(quant, cv2.COLOR_BGR2GRAY)

        # Create blank white canvas and draw simplified contours from quantized regions
        canvas = np.ones_like(quant_gray) * 255

        for thresh_val in [50, 100, 150, 200]:
            _, th = cv2.threshold(quant_gray, thresh_val, 255, cv2.THRESH_BINARY)
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > min_area:
                    eps = 0.02 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, eps, True)
                    cv2.drawContours(canvas, [approx], -1, 0, line_thickness)

        # Combine canvas contours with the edge map: keep lines where either has ink
        final = cv2.bitwise_and(canvas, edges)

        # Small cleanups
        final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))

        # Optionally invert colors if requested
        invert_colors = request.form.get('invert_colors', None)
        if str(invert_colors).lower() in ('1', 'true', 'yes', 'y'):
            final = cv2.bitwise_not(final)

        pil_img = Image.fromarray(final)
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 9000))
    app.run(debug=True, host='0.0.0.0', port=port)
