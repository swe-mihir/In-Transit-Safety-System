# 🚗 In-Transit Safety System

A real-time in-vehicle audio safety monitoring pipeline that classifies ambient audio into **NORMAL**, **RISKY**, or **DANGEROUS** categories using a multi-model AI architecture.

Built as a Semester 6 Mini Project — Third Year Engineering.

---

## Overview

The system captures live microphone audio from a smartphone browser via WebRTC, runs it through a hybrid ASR + sentiment + background sound analysis pipeline, and produces a fused safety decision every few seconds. Results are displayed on a live web dashboard.

```
Phone Mic (WebRTC)
      ↓
  server.py          — captures audio, saves 10s WAV chunks
      ↓
  audio_preprocessor.py  — mono, 16kHz resample, peak normalise
      ↓
  hybrid_asr.py      — Whisper transcription + Wav2Vec2 quality gate
      ↓
  ┌──────────────────────┐   ┌─────────────────────┐
  │  model_b_inference   │   │   yamnet_model.py   │  ← parallel
  │  (Hindi BERT)        │   │   (background SFX)  │
  │  Model A  wt=0.6     │   │   Model B  wt=0.4   │
  └──────────┬───────────┘   └──────────┬──────────┘
             └──────────────────────────┘
                         ↓
                     fusion.py         — weighted avg + safety-first escalation
                         ↓
               NORMAL / RISKY / DANGEROUS
                         ↓
               dashboard_server.py    — live browser UI on :5050
```

---

## Features

- **Real-time audio capture** via WebRTC from any smartphone browser — no app install needed
- **Hybrid Hindi ASR** — Whisper (transcription) + Wav2Vec2 (audio quality gate)
- **Dual-model safety classification**
  - Hindi BERT sentiment analysis on transcribed speech
  - YAMNet background sound classification (gunshots, screams, crashes, etc.)
- **Safety-first fusion** — weighted average with escalation logic (either model predicting danger is enough to escalate)
- **Live web dashboard** — real-time updates via WebSocket, model score bars, decision log, transcription view
- **Fully offline** after initial model download — no cloud API calls during inference

---

## Project Structure

```
FinalModel/
├── pipeline_master.py       # Master orchestrator — run this
├── server.py                # WebRTC audio capture server (fastrtc + Gradio)
├── audio_preprocessor.py   # Audio normalisation and resampling
├── hybrid_asr.py            # Whisper + Wav2Vec2 ASR pipeline
├── model_b_inference.py     # Hindi BERT sentiment inference (Model A)
├── yamnet_model.py          # YAMNet background sound classifier (Model B)
├── fusion.py                # Weighted fusion + safety-first escalation
├── dashboard_server.py      # Flask-SocketIO live dashboard backend
├── yamnet_class_map.csv     # YAMNet class labels (local copy)
├── cert.pem                 # SSL certificate for WebRTC (self-signed)
├── key.pem                  # SSL private key
├── sentiment/               # Hindi BERT model checkpoint
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── wav2vec2/                # Wav2Vec2 model checkpoint
│   ├── config.json
│   ├── model.safetensors
│   └── ...
└── Data capture/            # Auto-created — saved WAV chunks go here
```

---

## Requirements

**Python 3.10+** (tested on 3.12)

**System dependency:**
- [ffmpeg](https://ffmpeg.org/download.html) — must be on system PATH (used by Whisper internally)

**Python packages:**
```bash
pip install torch torchaudio transformers
pip install openai-whisper
pip install tensorflow tensorflow-hub
pip install fastrtc gradio
pip install flask flask-socketio
pip install librosa soundfile scipy numpy pandas
pip install cryptography
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/swe-mihir/In-Transit-Safety-System.git
cd In-Transit-Safety-System
git checkout final
```

### 2. Add model checkpoints

Place your model folders in the project root:

```
FinalModel/
├── sentiment/     ← Hindi BERT fine-tuned for safety classification
└── wav2vec2/      ← Wav2Vec2 for Hindi audio quality scoring
```

### 3. Download YAMNet class map (one time)

```bash
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv', 'yamnet_class_map.csv'); print('Done')"
```

### 4. Generate SSL certificate (required for WebRTC mic access)

```bash
python -c "
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import datetime, os

k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
n = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, u'localhost')])
c = (x509.CertificateBuilder()
     .subject_name(n).issuer_name(n)
     .public_key(k.public_key())
     .serial_number(x509.random_serial_number())
     .not_valid_before(datetime.datetime.utcnow())
     .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
     .sign(k, hashes.SHA256()))

open('key.pem','wb').write(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()))
open('cert.pem','wb').write(c.public_bytes(serialization.Encoding.PEM))
print('Done')
"
```

---

## Running

```bash
python pipeline_master.py
```

On startup you will see:

```
[Dashboard] Open http://localhost:5050 in your browser

  📦  Loading all models — please wait …
  ✅  All models loaded.

[Worker] Pipeline worker started — waiting for audio chunks …

  🎙️  Real-Time Audio Streaming Server
  Laptop : https://localhost:7860
  Phone  : https://x.x.x.x:7860
  Running on public URL: https://xxxxxx.gradio.live
```

**Step 1** — Open `http://localhost:5050` in your browser (the live dashboard).

**Step 2** — Open the `gradio.live` public URL on your **smartphone** (allows mic access without SSL warning).

**Step 3** — Allow microphone permission on your phone and start speaking. Every 10 seconds of audio triggers a full pipeline run and the dashboard updates live.

---

## How It Works

### Audio Capture (`server.py`)
WebRTC streams audio from the phone browser to the laptop at 20ms frames via `fastrtc`. Frames are accumulated and saved as WAV chunks every 10 seconds. Each saved path is pushed to a queue consumed by the pipeline worker.

### Preprocessing (`audio_preprocessor.py`)
Each chunk is converted to mono, resampled to 16 kHz using polyphase resampling (scipy), and peak-normalised to ±1.0.

### Hybrid ASR (`hybrid_asr.py`)
- **Whisper** transcribes the Hindi speech to Devanagari text
- **Wav2Vec2** scores audio quality (0.0–1.0). Chunks below the confidence threshold (0.4) are flagged — sentiment analysis is skipped but YAMNet still runs

### Parallel Inference
Two branches run simultaneously in a `ThreadPoolExecutor`:

**Model A — Hindi BERT** (`model_b_inference.py`): Fine-tuned `BertForSequenceClassification` on Hindi safety-labelled text. Outputs `{normal, risky, dangerous}` probability distribution.

**Model B — YAMNet** (`yamnet_model.py`): Google's audio event classifier (521 classes). Danger keywords (gunshot, explosion, crash) and risky keywords (scream, alarm) are mapped to the 3-class safety distribution.

### Fusion (`fusion.py`)
Weighted average of both model outputs (Model A: 0.6, Model B: 0.4) produces a single danger score. Safety-first escalation logic ensures that if **either** model predicts Risky or Dangerous, the final label cannot be downgraded below the more severe prediction.

| Score Range | Label     |
|-------------|-----------|
| 0.00 – 0.30 | NORMAL    |
| 0.30 – 0.70 | RISKY     |
| 0.70 – 1.00 | DANGEROUS |

### Dashboard (`dashboard_server.py`)
Flask-SocketIO server on port 5050. `pipeline_master` calls `push_decision()` after each chunk, which broadcasts the result to all connected browsers via WebSocket. The dashboard shows the current safety status, score ring, transcription, model probability bars, and a running decision log.

---

## Configuration

Key constants can be changed at the top of each file or via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_SIZE` | `small` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `MODEL_B_PATH` | `./sentiment` | Path to Hindi BERT checkpoint folder |
| `WAV2VEC_PATH` | `./wav2vec2` | Path to Wav2Vec2 checkpoint folder |
| `CHUNK_DURATION_S` | `10` | Seconds of audio per pipeline chunk (in `server.py`) |

---

## Known Issues & Notes

- On Windows, all model paths must use forward slashes — this is handled automatically via `os.path.abspath().replace("\\", "/")`
- Whisper uses ffmpeg internally as a subprocess. ffmpeg must be on the system PATH
- YAMNet is loaded from TensorFlow Hub on first run (cached locally after). The class map CSV must be downloaded separately for offline use
- The `gradio.live` tunnel is used for phone connectivity — this requires internet on the laptop but the actual audio inference is fully local
- TensorFlow deprecation warnings on startup are cosmetic and do not affect functionality

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     CAPTURE LAYER                           │
│  Phone Browser ──WebRTC──► fastrtc ──► RotatingWavWriter   │
└─────────────────────────────────┬───────────────────────────┘
                                  │ WAV path (queue)
┌─────────────────────────────────▼───────────────────────────┐
│                   PREPROCESSING LAYER                        │
│         mono conversion → 16kHz resample → peak norm        │
└─────────────────────────────────┬───────────────────────────┘
                                  │ float32 waveform
┌─────────────────────────────────▼───────────────────────────┐
│                       ASR LAYER                              │
│   Whisper (transcription)  +  Wav2Vec2 (quality gate)       │
└──────────────┬──────────────────────────────────────────────┘
               │ Hindi text (if quality ≥ 0.4)
    ┌──────────▼──────────┐        ┌──────────────────────┐
    │  Hindi BERT         │        │  YAMNet              │
    │  (Model A, w=0.6)   │        │  (Model B, w=0.4)    │
    │  sentiment on text  │        │  events on audio     │
    └──────────┬──────────┘        └──────────┬───────────┘
               └──────────────┬───────────────┘
                    ┌─────────▼──────────┐
                    │  Weighted Fusion   │
                    │  + Safety-First    │
                    │  Escalation        │
                    └─────────┬──────────┘
                              │
              ┌───────────────▼────────────────┐
              │   NORMAL / RISKY / DANGEROUS    │
              └───────────────┬────────────────┘
                              │ WebSocket
                    ┌─────────▼──────────┐
                    │  Live Dashboard    │
                    │  localhost:5050    │
                    └────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Audio capture | WebRTC via `fastrtc` + `aiortc` |
| UI transport | Gradio (signaling) + `gradio.live` tunnel |
| Speech recognition | OpenAI Whisper |
| Audio quality | Wav2Vec2 (HuggingFace Transformers) |
| Hindi NLP | BERT (`BertForSequenceClassification`) |
| Sound events | YAMNet (TensorFlow Hub) |
| Dashboard backend | Flask + Flask-SocketIO |
| Dashboard frontend | Vanilla JS + Socket.IO |
| Audio processing | librosa, scipy, numpy |

---

## License

Academic project — Semester 6 Mini Project, Third Year Engineering.
