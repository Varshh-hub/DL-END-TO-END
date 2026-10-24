# CardioSignal Web

## Structure
- `app.py` — Flask backend and PyTorch inference
- `templates/index.html` — HTML
- `static/style.css` — styling using `#F2DACE` and `#7C0E44`
- `static/script.js` — upload, waveform preview, API interaction
- `cardiosignal_cnn.pth` — add your trained checkpoint here

## Run locally
```bash
pip install -r requirements.txt
python app.py
```
Then open `http://127.0.0.1:5000`.

## Input
The demo expects a CSV containing exactly 216 numeric samples for one pre-segmented beat (one column or one row). It does not segment a full ECG recording into beats.

**Before using predictions:** Make sure the `ECG_CNN` class and `prepare_signal()` preprocessing in `app.py` exactly match your training notebook. The model architecture in this starter must be replaced if it differs from your actual checkpoint. The app is an educational research demo, not a medical diagnostic tool.
