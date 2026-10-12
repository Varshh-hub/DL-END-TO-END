import pickle
import tempfile
import traceback
from pathlib import Path

import librosa
import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Environmental Audio Classification System",
    page_icon="🎧",
    layout="centered",
)

MODEL_REPO_ID = "Varshhh6/environmental-audio-classifier"
MODEL_FILENAME = "bestmodel.pkl"
ALLOWED_EXTENSIONS = ["wav", "mp3", "ogg", "flac", "m4a"]
MAX_FILE_SIZE = 25 * 1024 * 1024


@st.cache_resource(show_spinner="Downloading and loading the model from Hugging Face...")
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model",
    )
    with open(model_path, "rb") as model_file:
        return pickle.load(model_file)


def extract_audio_features(file_path, trained_model):
    audio, sample_rate = librosa.load(
        file_path,
        sr=22050,
        mono=True,
    )

    if len(audio) == 0:
        raise ValueError("The uploaded audio file is empty.")

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    features_40 = mfcc_mean
    features_80 = np.concatenate([mfcc_mean, mfcc_std])

    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
    rms = librosa.feature.rms(y=audio)

    additional_features = np.array([
        np.mean(spectral_centroid),
        np.std(spectral_centroid),
        np.mean(spectral_bandwidth),
        np.mean(spectral_rolloff),
        np.mean(zero_crossing_rate),
        np.mean(rms),
    ])

    features_86 = np.concatenate([features_80, additional_features])
    expected_features = getattr(trained_model, "n_features_in_", None)

    candidates = {
        40: features_40,
        80: features_80,
        86: features_86,
    }

    if expected_features in candidates:
        selected_features = candidates[expected_features]
    elif expected_features is None:
        selected_features = features_80
    else:
        raise ValueError(
            f"The model expects {expected_features} input features. "
            "This app currently supports 40, 80, or 86 features. "
            "Update this function to exactly match the preprocessing in Model.ipynb."
        )

    return selected_features.reshape(1, -1)


st.title("🎧 Environmental Audio Classification System")
st.write(
    "Upload an environmental audio recording to classify its sound category."
)
st.caption("Supported formats: WAV, MP3, OGG, FLAC, M4A · Maximum size: 25 MB")

with st.expander("Model information"):
    st.write("Model repository: " + MODEL_REPO_ID)
    st.write("The large `bestmodel.pkl` file is downloaded from Hugging Face at runtime.")
    st.warning(
        "The feature extraction in this app must match the exact preprocessing "
        "used to train the model."
    )

try:
    model = load_model()
    st.success("Model loaded successfully.")
except Exception as exc:
    st.error("The model could not be downloaded or loaded.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.info(
        "Check that the Hugging Face repository is public and that the pickle "
        "file's training dependencies are included in requirements.txt."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=ALLOWED_EXTENSIONS,
)

if uploaded_file is not None:
    if uploaded_file.size == 0:
        st.error("The selected file is empty.")
        st.stop()

    if uploaded_file.size > MAX_FILE_SIZE:
        st.error("The file is too large. Please upload audio under 25 MB.")
        st.stop()

    st.audio(uploaded_file.getvalue())

    if st.button("Analyze audio", type="primary", use_container_width=True):
        suffix = Path(uploaded_file.name).suffix.lower()
        temporary_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temporary_path = temp_file.name

            with st.spinner("Extracting audio features and predicting..."):
                features = extract_audio_features(temporary_path, model)
                prediction = model.predict(features)[0]

                confidence = None
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(features)[0]
                    confidence = float(np.max(probabilities)) * 100

            st.subheader("Prediction result")
            st.success(f"Predicted class: {prediction}")

            if confidence is not None:
                st.metric("Model confidence", f"{confidence:.2f}%")

            st.caption(
                "If the model returns numeric class IDs, map them to the sound "
                "labels from your training dataset to display category names."
            )

        except Exception as exc:
            st.error("Audio prediction failed.")
            st.code(f"{type(exc).__name__}: {exc}")
            with st.expander("Technical details"):
                st.text(traceback.format_exc())

        finally:
            if temporary_path:
                try:
                    Path(temporary_path).unlink(missing_ok=True)
                except OSError:
                    pass

st.divider()
st.caption("Environmental Audio Classification System")
