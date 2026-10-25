import json
import os
import tempfile
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

SAMPLE_RATE = 16000
N_MFCC = 40
app = Flask(__name__)
CORS(app)

MODEL_PATH = "artifacts/speech_model.keras"
VOCAB_PATH = "artifacts/vocab.json"
model = None
char_to_id = None
id_to_char = None

def load_artifacts():
    global model, char_to_id, id_to_char
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VOCAB_PATH):
        return
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(VOCAB_PATH, encoding="utf-8") as f:
        vocab = json.load(f)
    char_to_id = vocab["char_to_id"]
    id_to_char = {int(k): v for k, v in vocab["id_to_char"].items()}

def features_from_audio(path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=12)
    if len(y) < int(0.15 * SAMPLE_RATE):
        raise ValueError("Audio segment is too short.")
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
                                n_fft=400, hop_length=160).T.astype("float32")
    mfcc = (mfcc - mfcc.mean(axis=0, keepdims=True)) / (mfcc.std(axis=0, keepdims=True) + 1e-5)
    return mfcc[None, ...]

def decode_greedy(probabilities):
    # CTC blank is the last class. Collapse repeats and remove blanks.
    blank_id = probabilities.shape[-1] - 1
    ids = np.argmax(probabilities[0], axis=-1)
    output, previous = [], None
    for idx in ids:
        idx = int(idx)
        if idx != previous and idx != blank_id:
            if idx in id_to_char:
                output.append(id_to_char[idx])
        previous = idx
    return "".join(output).strip()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if model is None:
        return jsonify({"error": "Model not found. Train it first using train.py."}), 503
    if "audio" not in request.files:
        return jsonify({"error": "No audio file received."}), 400
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            request.files["audio"].save(tmp.name)
            tmp_path = tmp.name
        feats = features_from_audio(tmp_path)
        probs = model.predict(feats, verbose=0)
        text = decode_greedy(probs)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": f"Could not transcribe this audio segment: {e}"}), 400
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

load_artifacts()

if __name__ == "__main__":
    if model is None:
        print("No trained model found. Run train.py first; the web page will show setup guidance.")
    app.run(host="127.0.0.1", port=5000, debug=True)
