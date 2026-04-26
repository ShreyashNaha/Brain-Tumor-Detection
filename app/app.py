"""
app.py
──────
Flask backend for Brain Tumor Detection Web App.

Routes:
  GET  /               → main page (index.html)
  GET  /api/models     → list available .pth files
  POST /api/analyze    → run full pipeline on uploaded image
  GET  /api/model-info → metadata about currently loaded model
"""

import os
import json
from flask import Flask, render_template, request, jsonify

from model_utils import list_models, load_model, DEVICE
from pipeline   import full_pipeline

# ── App Setup ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
UPLOAD_DIR  = os.path.join(BASE_DIR, 'uploads')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

ALLOWED_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}

# ── In-memory model cache (avoid reloading on every request) ──
_loaded_model      = None
_loaded_model_name = None
_loaded_model_meta = None


def get_model(model_filename: str):
    global _loaded_model, _loaded_model_name, _loaded_model_meta
    if model_filename != _loaded_model_name:
        model_path         = os.path.join(MODELS_DIR, model_filename)
        _loaded_model, _loaded_model_meta = load_model(model_path)
        _loaded_model_name = model_filename
        print(f"[INFO] Loaded model: {model_filename}")
    return _loaded_model, _loaded_model_meta


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/models', methods=['GET'])
def api_list_models():
    models = list_models(MODELS_DIR)
    return jsonify({'models': models})


@app.route('/api/model-info', methods=['GET'])
def api_model_info():
    filename = request.args.get('filename', '')
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    try:
        _, meta = get_model(filename)
        return jsonify({'meta': meta})
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    # ── Validate model selection ──
    model_filename = request.form.get('model_filename', '').strip()
    if not model_filename:
        return jsonify({'error': 'No model selected'}), 400

    available = list_models(MODELS_DIR)
    if model_filename not in available:
        return jsonify({'error': f'Model "{model_filename}" not found in models/'}), 404

    # ── Validate file upload ──
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({'error': f'Unsupported file type: {ext}'}), 400

    image_bytes = file.read()
    if len(image_bytes) == 0:
        return jsonify({'error': 'Uploaded file is empty'}), 400

    # ── Run pipeline ──
    try:
        model, meta = get_model(model_filename)
        results     = full_pipeline(model, image_bytes, DEVICE)
        results['model_meta'] = meta
        return jsonify({'success': True, 'data': results})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Pipeline error: {str(e)}'}), 500


# ── Dev server entry ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  🧠  Brain Tumor Detection — Flask App")
    print(f"  Models dir : {MODELS_DIR}")
    print(f"  Device     : {DEVICE}")
    print("  URL        : http://127.0.0.1:5000")
    print("=" * 55)

    models_found = list_models(MODELS_DIR)
    if not models_found:
        print("\n  ⚠️  No .pth files found in app/models/")
        print("      Train on Colab, download the .pth, and place it there.\n")
    else:
        print(f"\n  ✅  Models available: {models_found}\n")

    app.run(debug=True, host='127.0.0.1', port=5000)