# CardioSignal: ECG Heartbeat Classification and Analysis

A deep learning project that classifies ECG heartbeat segments as **Normal or Abnormal** using a 1D Convolutional Neural Network (CNN) implemented with PyTorch, integrated with a Flask-based web application.

> ⚠️ **Educational Purpose Only:** This project is developed strictly for educational and experimental purposes. Predictions may not always be accurate and must not be considered medical advice or a diagnosis. This application is not intended for clinical use or real-world medical decision-making. Never rely on this model to diagnose heart conditions or make medical decisions.

---

## Overview

**CardioSignal** explores the application of deep learning to electrocardiogram (ECG) heartbeat classification.

The system processes ECG signal segments and uses a trained 1D CNN to classify them into Normal or Abnormal categories. A Flask-based web interface allows users to submit ECG data in CSV format and view the model's prediction.

The project demonstrates an end-to-end deep learning workflow, including dataset acquisition, signal preprocessing, heartbeat segmentation, model training, evaluation, model integration, and web-based inference.

The predictions are experimental model outputs and may not accurately reflect a person's actual heart condition.

---

## Key Features

- ECG dataset acquisition using Python.
- ECG signal preprocessing and heartbeat segmentation.
- Fixed-length ECG input preparation.
- 1D CNN-based binary classification using PyTorch.
- Trained model integration with Flask.
- Web interface for uploading ECG data in CSV format.
- Normal and Abnormal classification results.
- Prediction probability and threshold-based classification, as implemented in the application.
- Model checkpoint loading for inference.
- Interactive web-based prediction workflow.

---

## Deep Learning Workflow

```text
       MIT-BIH ECG Dataset
               |
               v
       Dataset Acquisition
               |
               v
       ECG Signal Processing
               |
               v
      Heartbeat Segmentation
               |
               v
        Input Preparation
               |
               v
          1D CNN Model
               |
               v
         Model Training
               |
               v
        Model Evaluation
               |
               v
          Model Saving
               |
               v
        Flask Integration
               |
               v
        Web Application
               |
               v
     Normal / Abnormal Output
```

---

## Project Structure

```text
DL-END-TO-END/
│
└── 2. DL - ECG HEARTBEAT CLASSIFICATION AND ANALYSIS/
    │
    ├── Model.ipynb
    │
    └── web/
        ├── app.py
        ├── cardiosignal_cnn.pth
        ├── requirements.txt
        ├── README.md
        │
        ├── templates/
        │   └── index.html
        │
        └── static/
            ├── style.css
            └── script.js
```

*Note: Update the notebook filename or directory structure above if it differs from the files in your repository.*

---

## Dataset

This project uses the **MIT-BIH Arrhythmia Database** for ECG signal analysis and heartbeat classification.

The dataset can be downloaded programmatically using the Python `wfdb` library, as implemented in the training notebook.

The database contains annotated ECG recordings that support experimentation with heartbeat classification.

Before running the notebook, ensure that:

- You have a working internet connection for dataset acquisition.
- The required Python libraries are installed.
- Dataset downloading completes successfully.
- Sufficient disk space is available for the recordings and processed data.

Dataset labels and preprocessing choices affect the classes learned by the model. The project's binary classification output should not be interpreted as a clinical diagnosis.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Varshh-hub/DL-END-TO-END.git
```

### 2. Navigate to the ECG Project Directory

```bash
cd "DL-END-TO-END/2. DL - ECG HEARTBEAT CLASSIFICATION AND ANALYSIS"
```

### 3. Create a Virtual Environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 4. Install Training Dependencies

Install the libraries required by the notebook:

```bash
pip install numpy pandas matplotlib scikit-learn torch wfdb jupyter
```

Additional packages may be required depending on the notebook implementation.

### 5. Run the Training Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the ECG model notebook and execute its cells in order to download the dataset, preprocess the signals, prepare heartbeat segments, train the CNN, and evaluate its performance.

---

## Web Application Setup

### 1. Navigate to the Web Directory

From the ECG project directory, run:

```bash
cd web
```

### 2. Create and Activate a Virtual Environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Web Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify the Model Checkpoint

Ensure the following file exists in the `web/` directory:

```text
cardiosignal_cnn.pth
```

The checkpoint must match the CNN architecture and the checkpoint format expected by `app.py`. If the file is missing or incompatible, the model may fail to load.

### 5. Start the Flask Application

```bash
python app.py
```

Open the local application in your browser:

```text
http://127.0.0.1:5009
```

The configured port may differ if the application or environment variable changes it.

---

## Model Architecture

The project uses a **1D Convolutional Neural Network (CNN)** implemented in PyTorch.

The architecture includes:

- **Convolutional layers:** Learn patterns from one-dimensional ECG signal sequences.
- **Batch normalization:** Normalizes intermediate activations during training.
- **ReLU activation:** Introduces non-linearity.
- **Pooling layers:** Reduce intermediate sequence dimensions.
- **Dropout:** Helps reduce overfitting during training.
- **Adaptive average pooling:** Produces a fixed-size feature representation.
- **Fully connected layers:** Map learned features to the final classification output.

The web application is configured to process ECG input segments containing **216 numerical samples**.

The model produces a logit, which the application converts into a probability using the sigmoid function. The probability is compared with the configured threshold to generate the Normal or Abnormal classification.

These outputs reflect the model's learned patterns and are not guaranteed to be medically accurate.

---

## Web Application Workflow

1. Open the CardioSignal web interface.
2. Prepare ECG signal data in the CSV format accepted by the application.
3. Upload the CSV file.
4. The application validates the input and checks the sample count.
5. The input signal is normalized according to the preprocessing implemented in the application.
6. The trained CNN generates a prediction.
7. The application displays the supported classification result and prediction details.

### Input Requirements

- The application expects **216 numerical ECG samples**.
- The uploaded CSV must follow the layout accepted by the application.
- The signal should be compatible with the preprocessing assumptions used during training.
- Invalid or incompatible inputs may produce errors or unreliable predictions.

The application is intended for experimentation with prepared ECG data and not for clinical interpretation.

---

## Technologies Used

**Programming and Development**
- Python
- Jupyter Notebook

**Data Processing**
- NumPy
- Pandas
- Scikit-learn

**Deep Learning**
- PyTorch
- 1D Convolutional Neural Networks
- Binary Classification

**ECG Processing**
- WFDB
- MIT-BIH Arrhythmia Database
- ECG Signal Preprocessing

**Web Development**
- Flask
- HTML
- CSS
- JavaScript

---

## Educational Purpose and Disclaimer

**This project is strictly for educational, learning, and experimental purposes only.**

- The model's predictions may be inaccurate or incorrect.
- The application is not a certified or clinically validated medical device.
- Normal and Abnormal are model classification labels, not definitive assessments of heart health.
- Prediction probabilities do not necessarily represent clinical certainty.
- Model performance may vary with signal quality, preprocessing, data distribution, and recordings from different devices or populations.
- The application must not be used to diagnose heart conditions, recommend treatments, or determine whether medical care is necessary.
- Do not use this project for real-world patient monitoring or safety-critical medical decisions.

**Always consult a qualified healthcare professional for medical concerns.** Never delay seeking medical attention based on a prediction from this application.

---

## Limitations

- The model depends on the dataset and preprocessing used during training.
- The current application expects a fixed input length and a specific CSV format.
- Unseen ECG patterns may not be classified correctly.
- Prediction confidence does not guarantee prediction correctness.
- The model has not been established as suitable for clinical use.
- The current implementation may not support direct processing of raw ECG files from all devices.

---

## Future Enhancements

- Support additional ECG input formats.
- Add ECG waveform visualization.
- Improve input validation and error handling.
- Compare multiple CNN architectures.
- Evaluate performance on independent datasets.
- Enhance prediction-result visualization.
- Improve application usability for educational demonstrations.

These are potential enhancements and may not be available in the current version.

---

## Author

**Varsha A.**

AI & ML Graduate | Aspiring Data Scientist & Machine Learning Engineer

**Technical Interests:** Python | SQL | Machine Learning | Deep Learning

- **GitHub:** https://github.com/Varshh-hub
- **Portfolio:** https://varshh-hub.github.io/VARSHA---portfolio/

---

*CardioSignal is an educational deep learning project. Its predictions may not be accurate and must not be used as a substitute for professional medical evaluation.*
