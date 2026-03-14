"""
audio_preprocessor.py
----------------------
Real-time audio preprocessing module for a machine-learning pipeline.

Dependencies: numpy, scipy  (standard scientific stack — no extra install)
Optional:     librosa, soundfile  (used automatically when available)

Pipeline (per chunk)
--------------------
  1. Detect channel configuration  (mono / stereo)
  2. Stereo → mono  (average channels)
  3. Resample to target_sr if needed
  4. Peak-amplitude normalisation   (no filtering / noise removal)
  5. Return cleaned waveform

Input  : NumPy array — shape (N,) for mono  or  (2, N) / (N, 2) for stereo
Output : NumPy array — shape (N,) mono, float32, peak-normalised to ±1
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SR: int = 16_000       # downstream models expect 16 kHz mono
NORM_PEAK: float = 1.0        # target peak amplitude
NORM_EPSILON: float = 1e-8    # divide-by-zero guard on silent frames


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stereo_to_mono(audio: np.ndarray) -> np.ndarray:
    """Average two channels to produce a mono waveform.

    Parameters
    ----------
    audio : np.ndarray, shape (2, N)
        Channels-first stereo array (float32 or float64).

    Returns
    -------
    np.ndarray, shape (N,), float64
        Averaged mono waveform.  Averaging preserves the full dynamic range
        of both channels (including background / environmental sounds) without
        favouring or attenuating either channel.
    """
    return (audio[0].astype(np.float64) + audio[1].astype(np.float64)) * 0.5


def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resample using scipy.signal.resample_poly.

    Polyphase resampling is efficient and introduces minimal distortion,
    making it suitable for real-time buffers of 20–100 ms.

    Parameters
    ----------
    waveform : np.ndarray, shape (N,)
        Mono float waveform at *orig_sr*.
    orig_sr : int
    target_sr : int

    Returns
    -------
    np.ndarray, shape (M,), float32
        Resampled waveform where M ≈ N * target_sr / orig_sr.
    """
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
    """Scale waveform so that max(|x|) == peak.

    Rationale
    ---------
    Peak normalisation is the lightest-weight normalisation strategy.  It
    simply scales the entire chunk by a single scalar, so the *relative*
    amplitude relationship between all sounds (dialogue, footsteps, ambient
    noise …) is preserved exactly.  No clipping occurs because the output
    peak is exactly *peak* (≤ 1.0 by default).

    Silent / near-silent frames (max_abs ≤ epsilon) are returned unchanged to
    avoid amplifying pure noise to full-scale.

    Parameters
    ----------
    waveform : np.ndarray
        Mono input waveform, any float dtype.
    peak : float
        Desired peak amplitude (default 1.0).
    epsilon : float
        Silent-frame guard (default 1e-8).

    Returns
    -------
    np.ndarray, float32
    """
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
    """Preprocess one audio chunk for real-time ML inference.

    Steps executed in order
    -----------------------
    1. Validate input and canonicalise array shape.
    2. Detect channel count  (1 = mono, 2 = stereo).
    3. Convert stereo → mono by averaging channels.
    4. Resample to *target_sr* when *sample_rate* differs.
    5. Peak-normalise to ±*peak* (no clipping, no filtering).

    Parameters
    ----------
    audio : np.ndarray
        Raw waveform captured from a microphone or streaming buffer.
        Accepted shapes:

        * ``(N,)``      — mono, 1-D array
        * ``(2, N)``    — stereo, channels-first   (torchaudio convention)
        * ``(N, 2)``    — stereo, channels-last    (soundfile convention)
        * ``(1, N)`` / ``(N, 1)`` — single-channel 2-D array

    sample_rate : int
        Native sample rate of the incoming chunk (e.g., 44100, 48000, 16000).
    target_sr : int, optional
        Output sample rate expected by downstream models (default 16 000 Hz).
        Set equal to *sample_rate* to skip resampling.
    peak : float, optional
        Target peak amplitude after normalisation (default 1.0).
    epsilon : float, optional
        Silent-frame guard; frames with max|x| ≤ epsilon are not scaled
        (default 1e-8).

    Returns
    -------
    waveform : np.ndarray
        Mono float32 waveform, shape ``(N_out,)``.
    out_sr : int
        Output sample rate.  Always equals *target_sr*.

    Raises
    ------
    ValueError
        If *audio* has an unsupported shape or number of dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> chunk = np.random.randn(2, 4800).astype(np.float32)   # stereo 48 kHz
    >>> out, sr = preprocess_audio_chunk(chunk, 48000)
    >>> out.shape, sr
    ((1600,), 16000)
    """

    # ── 0. Working precision ────────────────────────────────────────────────
    audio = np.asarray(audio, dtype=np.float32)

    # ── 1. Canonicalise shape → determine n_channels ────────────────────────
    if audio.ndim == 1:
        n_channels = 1
        waveform: np.ndarray = audio

    elif audio.ndim == 2:
        rows, cols = audio.shape

        if rows == 2 and cols != 2:
            # (2, N) — channels-first stereo
            n_channels = 2
            waveform = audio

        elif cols == 2 and rows != 2:
            # (N, 2) — channels-last stereo; transpose to (2, N)
            n_channels = 2
            waveform = audio.T.copy()

        elif rows == 2 and cols == 2:
            # Ambiguous 2×2: treat as channels-first (2 channels, 2 samples)
            n_channels = 2
            waveform = audio

        elif rows == 1:
            # (1, N) — single-channel 2-D array
            n_channels = 1
            waveform = audio[0]

        elif cols == 1:
            # (N, 1) — single-channel 2-D array
            n_channels = 1
            waveform = audio[:, 0]

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

    # ── 2. Stereo → mono ────────────────────────────────────────────────────
    if n_channels == 2:
        waveform = _stereo_to_mono(waveform).astype(np.float32)

    # ── 3. Resample ─────────────────────────────────────────────────────────
    if sample_rate != target_sr:
        waveform = _resample(waveform, sample_rate, target_sr)

    # ── 4. Peak normalisation ────────────────────────────────────────────────
    waveform = _peak_normalise(waveform, peak=peak, epsilon=epsilon)

    return waveform, target_sr


def preprocess_stream(
    chunks: list[np.ndarray],
    sample_rate: int,
    **kwargs,
) -> list[np.ndarray]:
    """Apply :func:`preprocess_audio_chunk` to every chunk in a live stream.

    Parameters
    ----------
    chunks : list of np.ndarray
        Raw audio buffers captured from the microphone stream.
    sample_rate : int
        Native sample rate shared by all chunks.
    **kwargs
        Forwarded verbatim to :func:`preprocess_audio_chunk`.

    Returns
    -------
    list of np.ndarray
        Preprocessed mono float32 waveforms, one per input chunk.
    """
    return [
        preprocess_audio_chunk(chunk, sample_rate, **kwargs)[0]
        for chunk in chunks
    ]
