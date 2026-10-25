# Live Speech-to-Text with BiLSTM + CTC (starter project)

This is an educational prototype that trains a character-level BiLSTM/CTC speech recognizer from Common Voice audio and serves short audio segments through a browser.

## 1. Install
Use Python 3.10 or 3.11 in a virtual environment:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get a dataset
Download an English Common Voice dataset from Mozilla's Common Voice site. Extract it so you have:
- `clips/` containing the audio files
- a tab-separated metadata file such as `validated.tsv`, with `path` and `sentence` columns

Then run (adjust paths as needed):
```bash
python train.py --tsv /path/to/validated.tsv --clips /path/to/clips --limit 3000 --epochs 10
```
This small limit is just to test the pipeline; a useful recognizer normally needs much more data and training. The script writes `artifacts/speech_model.keras` and `artifacts/vocab.json`.

## 3. Start the web app
```bash
python app.py
```
Open http://127.0.0.1:5000 in Chrome, allow microphone access, then click **Start**.

## Important limitations
- This is a learning prototype, not a production-grade streaming ASR system.
- The demo records short chunks and transcribes each chunk. A bidirectional LSTM needs future context, so it is not a true zero-latency streaming model; chunk boundaries can affect accuracy.
- Training from scratch on only a few thousand clips will likely produce poor transcripts. Increase the dataset, training time, and compute for better results.
- The model is English character-level. It does not automatically support Tamil or other languages.
