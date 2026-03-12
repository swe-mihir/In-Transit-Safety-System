# 🎙 Real-Time Audio Streaming — WebRTC → Python ML Pipeline

Stream live microphone audio from any smartphone browser to a Python backend
over your local WiFi network using **fastrtc + FastAPI**. No native app needed.

---

## Architecture

```
[Smartphone Browser]
   └─ getUserMedia() → WebRTC audio track
         │  (WebRTC / UDP over WiFi)
         ▼
[Laptop — Python server]
   ├─ FastAPI  (HTTP signalling: /webrtc/offer)
   ├─ fastrtc  (WebRTC engine, receives audio frames)
   └─ ml_pipeline()  ← YOUR MODEL GOES HERE
         numpy float32 array @ 16 000 Hz mono
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the server

```bash
python server.py
```

You'll see output like:
```
============================================================
  🎙️  Real-Time Audio Streaming Server
============================================================
  Laptop URL  : http://localhost:8000
  Phone URL   : http://192.168.1.42:8000   ← open on smartphone
============================================================
  Waiting for smartphone connection …
```

### 3. Open the client on your smartphone

- Make sure your phone and laptop are on the **same WiFi network**
- Open the **Phone URL** printed above in Safari or Chrome
- Tap **▶ Start Streaming**
- Grant microphone permission when prompted

### 4. Watch frames arrive on the server console

```
✅  First audio frame received — streaming is live!

[14:23:05]  samples=  320  duration= 20.0 ms  RMS=0.00312
[14:23:05]  samples=  320  duration= 20.0 ms  RMS=0.00287
[14:23:05]  samples=  320  duration= 20.0 ms  RMS=0.00401
...
```

---

## Plugging in Your ML Model

Edit `server.py`, find `ml_pipeline()`, and replace the placeholder:

```python
def ml_pipeline(audio_mono_16k: np.ndarray, sample_rate: int = 16000) -> None:
    # ── Whisper transcription example ──
    import whisper
    model = whisper.load_model("base")   # load once outside this fn in practice
    result = model.transcribe(audio_mono_16k, fp16=False)
    print(result["text"])

    # ── ONNX inference example ──
    # output = ort_session.run(None, {"input": audio_mono_16k[np.newaxis, :]})

    # ── Feature extraction example ──
    # mfcc = librosa.feature.mfcc(y=audio_mono_16k, sr=sample_rate)
```

**Tips:**
- Move heavy model loading (`load_model`, `ort.InferenceSession`) to module level,
  not inside `ml_pipeline()`.
- For low-latency inference use ONNX Runtime or TorchScript.
- Buffer multiple frames before inference if your model needs longer context:
  add a `collections.deque` in `AudioHandler.__init__`.

---

## File Reference

| File              | Purpose                                      |
|-------------------|----------------------------------------------|
| `server.py`       | FastAPI + fastrtc server, audio handler      |
| `client.html`     | Served to the smartphone browser             |
| `requirements.txt`| Python dependencies                          |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Phone can't reach the server URL | Check both devices are on the same WiFi; check firewall allows port 8000 |
| `getUserMedia` blocked | Use Chrome on Android or Safari on iOS; HTTPS not required on LAN |
| No frames in console | Verify ICE connection in browser log; restart server |
| High latency | Use wired ethernet on laptop; close other apps on phone |
| Resampling quality poor | `pip install scipy` and use `scipy.signal.resample` instead of the linear interpolation fallback |

---

## Tested With

- Python 3.11 / 3.12
- fastrtc 0.0.24+
- Chrome 124+ (Android), Safari 17+ (iOS)
- Ubuntu 22.04 / macOS 14 / Windows 11
