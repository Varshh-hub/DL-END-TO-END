
import os
import pickle
import traceback
from pathlib import Path

import librosa
import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

app = Flask(
    __name__,
    template_folder=".",
    static_folder=".",
    static_url_path="/static"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "bestmodel.pkl"
UPLOAD_DIR = BASE_DIR / "uploads"

ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "flac", "m4a"}

app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
UPLOAD_DIR.mkdir(exist_ok=True)

model = None
model_error = None

try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
except Exception as exc:
    model_error = str(exc)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def extract_audio_features(file_path):
    audio, sample_rate = librosa.load(
        file_path,
        sr=22050,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("The uploaded audio file is empty.")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    features_40 = mfcc_mean
    features_80 = np.concatenate([mfcc_mean, mfcc_std])

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sample_rate
    )
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio, sr=sample_rate
    )
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=sample_rate
    )
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
    rms = librosa.feature.rms(y=audio)

    additional_features = np.array([
        np.mean(spectral_centroid),
        np.std(spectral_centroid),
        np.mean(spectral_bandwidth),
        np.mean(spectral_rolloff),
        np.mean(zero_crossing_rate),
        np.mean(rms)
    ])

    features_86 = np.concatenate([
        features_80,
        additional_features
    ])

    expected_features = getattr(model, "n_features_in_", None)

    candidates = {
        40: features_40,
        80: features_80,
        86: features_86
    }

    if expected_features in candidates:
        selected_features = candidates[expected_features]
    elif expected_features is None:
        selected_features = features_80
    else:
        raise ValueError(
            f"The model expects {expected_features} features, but this "
            "app currently supports 40, 80, or 86 feature vectors. "
            "Update extract_audio_features() to match your notebook."
        )

    return selected_features.reshape(1, -1)


def predict_audio(file_path):
    if model is None:
        raise RuntimeError(
            "The model could not be loaded. Check that bestmodel.pkl "
            "exists in the parent project folder."
        )

    features = extract_audio_features(file_path)

    prediction = model.predict(features)[0]

    if isinstance(prediction, (np.integer, int)):
        predicted_label = str(prediction)
    else:
        predicted_label = str(prediction)

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = round(float(np.max(probabilities)) * 100, 2)

    return {
        "prediction": predicted_label,
        "confidence": confidence
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def classify_audio():
    if model is None:
        return jsonify({
            "error": "Model loading failed. Check bestmodel.pkl and its dependencies.",
            "details": model_error
        }), 500

    if "audio" not in request.files:
        return jsonify({"error": "Please select an audio file."}), 400

    audio_file = request.files["audio"]

    if not audio_file or not audio_file.filename:
        return jsonify({"error": "Please select an audio file."}), 400

    if not allowed_file(audio_file.filename):
        return jsonify({
            "error": "Unsupported file type. Use WAV, MP3, OGG, FLAC, or M4A."
        }), 400

    filename = secure_filename(audio_file.filename)
    file_path = UPLOAD_DIR / filename

    try:
        audio_file.save(file_path)
        result = predict_audio(str(file_path))
        return jsonify(result)

    except Exception as exc:
        return jsonify({
            "error": str(exc)
        }), 500

    finally:
        if file_path.exists():
            file_path.unlink()


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({
        "error": "The file is too large. Please upload audio under 25 MB."
    }), 413


if __name__ == "__main__":
    app.run(debug=True)