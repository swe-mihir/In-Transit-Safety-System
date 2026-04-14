"""
audio_preprocessor.py
----------------------
Real-time audio preprocessing module for the In-Transit Safety pipeline.

Pipeline (per chunk)
--------------------
  1. Detect channel configuration  (mono / stereo)
  2. Stereo → mono  (average channels)
  3. Resample to target_sr if needed  (polyphase, scipy)
  4. Peak-amplitude normalisation
  5. Return cleaned float32 waveform

Public API
----------
  preprocess_audio_chunk(audio, sample_rate, ...)  → (waveform, out_sr)
  preprocess_from_file(path, ...)                  → (waveform, out_sr)   ← NEW
  preprocess_stream(chunks, sample_rate, ...)      → list[np.ndarray]
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SR: int    = 16_000
NORM_PEAK: float  = 1.0
NORM_EPSILON: float = 1e-8


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stereo_to_mono(audio: np.ndarray) -> np.ndarray:
    """Average channels-first stereo (2, N) → mono (N,)."""
    return (audio[0].astype(np.float64) + audio[1].astype(np.float64)) * 0.5


def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resample via scipy.signal.resample_poly."""
    if orig_sr == target_sr:
        return waveform.astype(np.float32)
    g    = gcd(orig_sr, target_sr)
    up   = target_sr // g
    down = orig_sr   // g
    return resample_poly(waveform.astype(np.float64), up, down).astype(np.float32)


def _peak_normalise(
    waveform: np.ndarray,
    peak: float = NORM_PEAK,
    epsilon: float = NORM_EPSILON,
) -> np.ndarray:
    """Scale so max(|x|) == peak; silent frames are left untouched."""
    max_abs = float(np.max(np.abs(waveform)))
    if max_abs > epsilon:
        waveform = waveform * (peak / max_abs)
    return waveform.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_audio_chunk(
    audio: np.ndarray,
    sample_rate: int,
    *,
    target_sr: int = TARGET_SR,
    peak: float = NORM_PEAK,
    epsilon: float = NORM_EPSILON,
) -> tuple[np.ndarray, int]:
    """Preprocess one raw audio chunk for ML inference.

    Accepts shapes: (N,), (2,N), (N,2), (1,N), (N,1).
    Returns: (mono float32 waveform, target_sr).
    """
    audio = np.asarray(audio, dtype=np.float32)

    # ── Canonicalise shape ───────────────────────────────────────────────────
    if audio.ndim == 1:
        n_channels = 1
        waveform: np.ndarray = audio

    elif audio.ndim == 2:
        rows, cols = audio.shape

        if rows == 2 and cols != 2:
            n_channels, waveform = 2, audio
        elif cols == 2 and rows != 2:
            n_channels, waveform = 2, audio.T.copy()
        elif rows == 2 and cols == 2:
            n_channels, waveform = 2, audio          # ambiguous 2×2 → channels-first
        elif rows == 1:
            n_channels, waveform = 1, audio[0]
        elif cols == 1:
            n_channels, waveform = 1, audio[:, 0]
        else:
            raise ValueError(
                f"Unsupported 2-D audio shape {audio.shape}. "
                "Expected (N,), (2, N), or (N, 2)."
            )
    else:
        raise ValueError(
            f"Audio must be 1-D or 2-D, got {audio.ndim}-D array "
            f"with shape {audio.shape}."
        )

    if n_channels == 2:
        waveform = _stereo_to_mono(waveform).astype(np.float32)

    if sample_rate != target_sr:
        waveform = _resample(waveform, sample_rate, target_sr)

    waveform = _peak_normalise(waveform, peak=peak, epsilon=epsilon)
    return waveform, target_sr


def preprocess_from_file(
    audio_path: str,
    *,
    target_sr: int = TARGET_SR,
    **kwargs,
) -> tuple[np.ndarray, int]:
    """Load a WAV/audio file and run the full preprocessing pipeline.

    Uses librosa for loading (handles most formats + resampling hint).
    Falls back to soundfile if librosa is unavailable.

    Returns: (mono float32 waveform, target_sr).
    """
    try:
        import librosa
        audio, sr = librosa.load(audio_path, sr=None, mono=False)
    except ImportError:
        import soundfile as sf
        audio, sr = sf.read(audio_path, always_2d=False)
        audio = audio.T if audio.ndim == 2 else audio

    return preprocess_audio_chunk(audio, sr, target_sr=target_sr, **kwargs)


def preprocess_stream(
    chunks: list[np.ndarray],
    sample_rate: int,
    **kwargs,
) -> list[np.ndarray]:
    """Apply preprocess_audio_chunk to every chunk in a live stream."""
    return [
        preprocess_audio_chunk(chunk, sample_rate, **kwargs)[0]
        for chunk in chunks
    ]
