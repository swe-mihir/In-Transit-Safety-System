"""
Real-Time Audio Streaming Server
=================================
Compatible with fastrtc 0.0.34 (Gradio-based).

fastrtc 0.0.34 uses Gradio as its transport layer. The Stream object
launches a Gradio app that handles WebRTC signalling automatically.
The phone connects via the Gradio UI served at the printed URL.

Usage
-----
1. pip install fastrtc fastapi uvicorn numpy sounddevice
   pip install "fastrtc[vad]"   # optional, only if using ReplyOnPause

2. python server.py

3. Open the printed URL on your phone browser (same WiFi).
   No cert needed — Gradio handles the UI over plain HTTP on LAN.
"""

import time
import wave
import threading
import numpy as np
from fastrtc import Stream, ReplyOnPause

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PLAYBACK_ENABLED = False 
SAVE_WAV_ENABLED = True
WAV_OUTPUT_FILE  = "Data capture//recorded_audio.wav"
TARGET_SR        = 16_000

# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------

_sd_stream = None
_sd_lock   = threading.Lock()

def _get_playback_stream():
    global _sd_stream
    if _sd_stream is not None:
        return _sd_stream
    with _sd_lock:
        if _sd_stream is None:
            try:
                import sounddevice as sd
                _sd_stream = sd.OutputStream(
                    samplerate=TARGET_SR, channels=1,
                    dtype="float32", blocksize=0,
                )
                _sd_stream.start()
                print("🔊  Playback stream opened.")
            except Exception as e:
                print(f"⚠️  Playback unavailable: {e}")
                _sd_stream = False
    return _sd_stream

# ---------------------------------------------------------------------------
# WAV writer
# ---------------------------------------------------------------------------

class WavWriter:
    def __init__(self, path, sr):
        self._path, self._sr = path, sr
        self._frames, self._lock = [], threading.Lock()

    def write(self, audio: np.ndarray):
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        with self._lock:
            self._frames.append(pcm.tobytes())

    def save(self):
        with self._lock:
            data = b"".join(self._frames)
        if not data:
            print("⚠️  No audio recorded.")
            return
        with wave.open(self._path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._sr)
            wf.writeframes(data)
        print(f"\n💾  Saved {len(data)/2/self._sr:.1f}s → '{self._path}'")

_wav_writer = WavWriter(WAV_OUTPUT_FILE, TARGET_SR) if SAVE_WAV_ENABLED else None

# ---------------------------------------------------------------------------
# ML pipeline — plug your model in here
# ---------------------------------------------------------------------------

_frame_count = 0
_first_frame = True

def ml_pipeline(audio: np.ndarray, sr: int = TARGET_SR):
    global _frame_count, _first_frame
    _frame_count += 1

    if _first_frame:
        print("\n✅  First audio frame received — pipeline is live!\n")
        _first_frame = False

    if PLAYBACK_ENABLED:
        s = _get_playback_stream()
        if s:
            try:
                s.write(audio.reshape(-1, 1))
            except Exception:
                pass

    if SAVE_WAV_ENABLED and _wav_writer:
        _wav_writer.write(audio)

    rms = float(np.sqrt(np.mean(audio ** 2)))
    print(f"[{time.strftime('%H:%M:%S')}] frame={_frame_count:5d}  "
          f"samples={len(audio):5d}  "
          f"dur={len(audio)/sr*1000:.1f}ms  "
          f"RMS={rms:.5f}")

    # ── your model here ──
    # result = model.infer(audio)


# ---------------------------------------------------------------------------
# fastrtc 0.0.34 handler
# ReplyOnPause fires after each pause in speech.
# The handler receives (sample_rate, audio_array) and must yield audio back.
# ---------------------------------------------------------------------------

def audio_handler(audio: tuple[int, np.ndarray]):
    sample_rate, frames = audio

    # mono float32
    if frames.ndim == 2:
        mono = frames.mean(axis=0).astype(np.float32)
    else:
        mono = frames.astype(np.float32)

    # resample to 16 kHz if needed
    if sample_rate != TARGET_SR:
        new_len = int(len(mono) * TARGET_SR / sample_rate)
        mono = np.interp(
            np.linspace(0, len(mono) - 1, new_len),
            np.arange(len(mono)),
            mono,
        ).astype(np.float32)

    ml_pipeline(mono)

    # yield silence back to keep the stream open
    silence = np.zeros_like(frames)
    yield (sample_rate, silence)


# ---------------------------------------------------------------------------
# Stream — launches Gradio UI with built-in WebRTC support
# ---------------------------------------------------------------------------

stream = Stream(
    handler=ReplyOnPause(audio_handler),
    modality="audio",
    mode="send-receive",
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

import atexit
import signal
import sys

def _save_and_exit(*args):
    """Flush WAV to disk then exit — triggered by Ctrl+C or process exit."""
    if SAVE_WAV_ENABLED and _wav_writer:
        _wav_writer.save()
    sys.exit(0)

atexit.register(_save_and_exit)
signal.signal(signal.SIGINT,  _save_and_exit)
signal.signal(signal.SIGTERM, _save_and_exit)


if __name__ == "__main__":
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
    except Exception:
        local_ip = "127.0.0.1"
    finally:
        s.close()

    port = 7860  # default Gradio port

    # ── SSL cert check ────────────────────────────────────────────────────
    import os
    cert_file, key_file = "Data capture//cert.pem", "Data capture//key.pem"
    if not (os.path.exists(cert_file) and os.path.exists(key_file)):
        print("\n  ERROR: cert.pem / key.pem not found.")
        print("  Run:  python generate_cert.py\n")
        raise SystemExit(1)

    print("=" * 60)
    print("  🎙️  Real-Time Audio Streaming Server")
    print("  (fastrtc 0.0.34 / Gradio transport)")
    print("=" * 60)
    print(f"  Laptop : https://localhost:{port}")
    print(f"  Phone  : https://{local_ip}:{port}   <- open on smartphone")
    print("=" * 60)
    print(f"  Playback : {'ON' if PLAYBACK_ENABLED else 'OFF'}")
    print(f"  Save WAV : {'ON -> ' + WAV_OUTPUT_FILE if SAVE_WAV_ENABLED else 'OFF'}")
    print("=" * 60)
    print("  Browser will warn 'connection not private'.")
    print("  Tap Advanced -> Proceed to continue.")
    print("  Waiting for connection...")
    print()
    if SAVE_WAV_ENABLED:
        print(f"  💾  Audio will be saved to '{WAV_OUTPUT_FILE}'")
        print("      Press Ctrl+C in this terminal to stop and save the file.")
    print()

    stream.ui.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        ssl_certfile=cert_file,
        ssl_keyfile=key_file,
        ssl_verify=False,   # self-signed cert — skip chain verification
    )