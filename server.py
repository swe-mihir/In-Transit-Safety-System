"""
server.py — Real-Time Audio Streaming Server
"""

import os
import queue
import signal
import socket
import sys
import threading
import time
import wave

import numpy as np
from fastrtc import Stream, StreamHandler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PLAYBACK_ENABLED = False
CHUNK_DURATION_S = 5
WAV_OUTPUT_DIR   = "Data capture"
TARGET_SR        = 16_000

# ---------------------------------------------------------------------------
# Public queue
# ---------------------------------------------------------------------------

audio_chunk_queue: queue.Queue[str] = queue.Queue()

# ---------------------------------------------------------------------------
# Rotating WAV writer
# ---------------------------------------------------------------------------

class RotatingWavWriter:
    def __init__(self, output_dir: str, sr: int, chunk_duration_s: int):
        self._dir           = output_dir
        self._sr            = sr
        self._chunk_samples = sr * chunk_duration_s
        self._frames: list  = []
        self._sample_count  = 0
        self._lock          = threading.Lock()
        self._chunk_index   = 0
        os.makedirs(output_dir, exist_ok=True)

    def write(self, audio: np.ndarray):
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        with self._lock:
            self._frames.append(pcm)
            self._sample_count += len(pcm)
            if self._sample_count >= self._chunk_samples:
                self._rotate()

    def _rotate(self):
        data = np.concatenate(self._frames) if self._frames else np.array([], dtype=np.int16)
        self._frames       = []
        self._sample_count = 0
        self._chunk_index += 1
        if len(data) == 0:
            return
        path = self._make_path()
        self._save_wav(path, data)
        audio_chunk_queue.put(path)
        print(f"\n💾  Chunk saved → '{path}'  "
              f"({len(data) / self._sr:.1f}s)  "
              f"[queue size: {audio_chunk_queue.qsize()}]\n")

    def flush(self):
        with self._lock:
            if self._sample_count > 0:
                print("\n⏹️  Flushing final partial chunk…")
                self._rotate()

    def _make_path(self) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(self._dir, f"chunk_{self._chunk_index:04d}_{ts}.wav")

    @staticmethod
    def _save_wav(path: str, data: np.ndarray):
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_SR)
            wf.writeframes(data.tobytes())


_wav_writer = RotatingWavWriter(WAV_OUTPUT_DIR, TARGET_SR, CHUNK_DURATION_S)

# ---------------------------------------------------------------------------
# StreamHandler subclass — required by fastrtc send-receive mode
# ---------------------------------------------------------------------------

class AudioCaptureHandler(StreamHandler):
    """
    Receives every audio frame from the browser microphone,
    normalises it, resamples to 16 kHz, and feeds the rotating WAV writer.
    """

    def __init__(self):
        super().__init__()
        self._frame_count = 0
        self._first_frame = True

    # Called by fastrtc for every incoming audio frame
    def receive(self, frame: tuple[int, np.ndarray]) -> None:
        sample_rate, frames = frame
        self._frame_count += 1

        if self._first_frame:
            print("\n✅  First audio frame received — pipeline is live!\n")
            self._first_frame = False

        # ── Mono ────────────────────────────────────────────────────────────
        if frames.ndim == 2:
            mono = frames.mean(axis=0).astype(np.float32) \
                   if frames.shape[0] < frames.shape[1] \
                   else frames.mean(axis=1).astype(np.float32)
        else:
            mono = frames.astype(np.float32)

        # ── Normalise int16 → float32 [-1, 1] if needed ─────────────────────
        max_val = np.max(np.abs(mono))
        if max_val > 1.0:
            mono = mono / 32768.0

        # ── Resample to 16 kHz if needed ────────────────────────────────────
        if sample_rate != TARGET_SR:
            new_len = int(len(mono) * TARGET_SR / sample_rate)
            mono = np.interp(
                np.linspace(0, len(mono) - 1, new_len),
                np.arange(len(mono)),
                mono,
            ).astype(np.float32)

        _wav_writer.write(mono)

        rms = float(np.sqrt(np.mean(mono ** 2)))
        print(f"[{time.strftime('%H:%M:%S')}] frame={self._frame_count:5d}  "
              f"samples={len(mono):5d}  "
              f"dur={len(mono)/TARGET_SR*1000:.1f}ms  "
              f"RMS={rms:.5f}")

    # Required by fastrtc — returns a fresh handler instance for each connection
    def copy(self):
        return AudioCaptureHandler()

    # Must be implemented — returns silence to keep the WebRTC stream alive
    def emit(self) -> tuple[int, np.ndarray]:
        return (TARGET_SR, np.zeros(TARGET_SR // 10, dtype=np.float32))


# ---------------------------------------------------------------------------
# Stream object
# ---------------------------------------------------------------------------

stream = Stream(
    handler=AudioCaptureHandler(),
    modality="audio",
    mode="send-receive",
    rtc_configuration={
        "iceServers": [
            {"urls": "stun:stun.l.google.com:19302"},
            {"urls": "stun:stun1.l.google.com:19302"},
        ]
    },
)


# ---------------------------------------------------------------------------
# Public start function
# ---------------------------------------------------------------------------

def start_server(port: int = 7860,
                 cert_file: str = "cert.pem",
                 key_file:  str = "key.pem"):
    import atexit
    atexit.register(_wav_writer.flush)
    signal.signal(signal.SIGINT,  lambda *_: (_wav_writer.flush(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda *_: (_wav_writer.flush(), sys.exit(0)))

    if not (os.path.exists(cert_file) and os.path.exists(key_file)):
        print("\n  ERROR: cert.pem / key.pem not found.")
        print("  Run:  python generate_cert.py\n")
        raise SystemExit(1)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
    except Exception:
        local_ip = "127.0.0.1"
    finally:
        s.close()

    print("=" * 60)
    print("  🎙️  Real-Time Audio Streaming Server")
    print(f"  Laptop : https://localhost:{port}")
    print(f"  Phone  : https://{local_ip}:{port}")
    print(f"  Chunks : {CHUNK_DURATION_S}s → {WAV_OUTPUT_DIR}/")
    print("=" * 60)

    stream.ui.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=True,
        ssl_certfile=cert_file,
        ssl_keyfile=key_file,
        ssl_verify=False,
    )


if __name__ == "__main__":
    start_server()