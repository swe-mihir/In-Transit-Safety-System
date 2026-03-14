"""
test_wav_file.py
----------------
Load a real .wav file, pass it through the audio preprocessor, and
print a detailed before/after report.

Usage
-----
    # Basic — file path as positional argument:
    python test_wav_file.py path/to/your_audio.wav

    # Optional: override the output sample-rate (default 16 000 Hz):
    python test_wav_file.py path/to/your_audio.wav --target-sr 16000

    # Save the preprocessed audio to a new .wav file:
    python test_wav_file.py path/to/your_audio.wav --save

Dependencies
------------
    numpy, scipy  (already in the standard scientific stack — no pip install needed)
    audio_preprocessor.py  (must be in the same directory as this script)
"""

from __future__ import annotations

import argparse
import os
import sys
import wave

import numpy as np
import scipy.io.wavfile as wav_io

# ── locate audio_preprocessor.py in the same directory as this script ──────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_preprocessor import preprocess_audio_chunk, TARGET_SR

SEP  = "─" * 64
SEP2 = "═" * 64


# ---------------------------------------------------------------------------
# WAV reader  (pure stdlib + scipy — no soundfile / librosa required)
# ---------------------------------------------------------------------------

def read_wav(path: str) -> tuple[np.ndarray, int, int, int]:
    """Read a WAV file and return raw audio as a float32 NumPy array.

    Parameters
    ----------
    path : str
        Path to the .wav file.

    Returns
    -------
    audio : np.ndarray
        Float32 waveform.
        Shape ``(N,)`` for mono, ``(N, 2)`` for stereo.
    sample_rate : int
    n_channels : int
    bit_depth : int
    """
    # Use stdlib `wave` only to read metadata (reliable for all PCM WAVs)
    with wave.open(path, "rb") as wf:
        n_channels  = wf.getnchannels()
        sample_rate = wf.getframerate()
        bit_depth   = wf.getsampwidth() * 8   # bytes → bits

    # Use scipy for the actual sample data (handles 8/16/24/32-bit PCM)
    sr, data = wav_io.read(path)

    # Normalise integer PCM types to float32 in [-1.0, +1.0]
    if data.dtype == np.uint8:           # 8-bit unsigned  [0, 255]
        audio = (data.astype(np.float32) - 128.0) / 128.0
    elif data.dtype == np.int16:         # 16-bit signed
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:         # 32-bit signed
        audio = data.astype(np.float32) / 2147483648.0
    elif np.issubdtype(data.dtype, np.floating):
        audio = data.astype(np.float32) # already float (e.g. 32-bit float WAV)
    else:
        raise ValueError(f"Unsupported WAV sample dtype: {data.dtype}")

    return audio, sample_rate, n_channels, bit_depth


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _duration_str(n_samples: int, sr: int) -> str:
    secs = n_samples / sr
    m, s = divmod(secs, 60)
    return f"{int(m):02d}:{s:06.3f}  ({secs:.4f} s)"


def print_waveform_stats(label: str, audio: np.ndarray, sr: int) -> None:
    mono = audio if audio.ndim == 1 else None   # only compute per-channel for mono

    print(f"\n  {label}")
    print(f"    Shape          : {audio.shape}")
    print(f"    dtype          : {audio.dtype}")
    print(f"    Channels       : {1 if audio.ndim == 1 else audio.shape[1]}")
    print(f"    Sample rate    : {sr:,} Hz")

    n = audio.shape[0]
    print(f"    Sample count   : {n:,}")
    print(f"    Duration       : {_duration_str(n, sr)}")

    flat = audio.flatten()
    print(f"    Min value      : {flat.min():.6f}")
    print(f"    Max value      : {flat.max():.6f}")
    print(f"    Max |amplitude|: {np.max(np.abs(flat)):.6f}")
    print(f"    RMS            : {np.sqrt(np.mean(flat ** 2)):.6f}")
    print(f"    Mean           : {flat.mean():.6f}")
    print(f"    Std dev        : {flat.std():.6f}")

    if audio.ndim == 2:
        for ch in range(audio.shape[1]):
            ch_data = audio[:, ch]
            print(f"    Channel {ch+1} max |x|: {np.max(np.abs(ch_data)):.6f}  "
                  f"RMS: {np.sqrt(np.mean(ch_data**2)):.6f}")


# ---------------------------------------------------------------------------
# WAV writer
# ---------------------------------------------------------------------------

def save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    """Write a float32 mono waveform as a 16-bit PCM WAV file."""
    # Clip to [-1, 1] before converting to int16
    clipped = np.clip(audio, -1.0, 1.0)
    int16   = (clipped * 32767).astype(np.int16)
    wav_io.write(path, sr, int16)


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test audio_preprocessor.py on a real .wav file."
    )
    parser.add_argument(
        "wav_file",
        help="Path to the input .wav file."
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=TARGET_SR,
        metavar="HZ",
        help=f"Target sample rate for the preprocessor (default: {TARGET_SR})."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the preprocessed audio as a new .wav file next to the input."
    )
    args = parser.parse_args()

    wav_path = os.path.abspath(args.wav_file)

    # ── 1. Validate file ────────────────────────────────────────────────────
    if not os.path.isfile(wav_path):
        print(f"\n  ERROR: File not found → {wav_path}", file=sys.stderr)
        sys.exit(1)

    file_size_kb = os.path.getsize(wav_path) / 1024

    print(SEP2)
    print("  WAV File Preprocessor Test")
    print(SEP2)
    print(f"\n  File   : {wav_path}")
    print(f"  Size   : {file_size_kb:.1f} KB")

    # ── 2. Read WAV ─────────────────────────────────────────────────────────
    try:
        audio_raw, sr, n_channels, bit_depth = read_wav(wav_path)
    except Exception as exc:
        print(f"\n  ERROR reading WAV: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  Format : {bit_depth}-bit PCM, {n_channels}-channel, {sr:,} Hz")
    print(SEP)

    print_waveform_stats("BEFORE preprocessing (raw input)", audio_raw, sr)

    # ── 3. Preprocess ───────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  Running preprocess_audio_chunk()  →  target_sr={args.target_sr:,} Hz")
    print(SEP)

    try:
        audio_out, out_sr = preprocess_audio_chunk(
            audio_raw, sr, target_sr=args.target_sr
        )
    except Exception as exc:
        print(f"\n  ERROR during preprocessing: {exc}", file=sys.stderr)
        sys.exit(1)

    print_waveform_stats("AFTER preprocessing (output)", audio_out, out_sr)

    # ── 4. Validation checks ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Validation")
    print(SEP)

    checks = {
        "Output is 1-D (mono)"          : audio_out.ndim == 1,
        "Output dtype is float32"        : audio_out.dtype == np.float32,
        f"Output sample rate = {out_sr:,} Hz": out_sr == args.target_sr,
        "Peak amplitude ≤ 1.0"           : float(np.max(np.abs(audio_out))) <= 1.0 + 1e-5,
        "Peak amplitude ≈ 1.0 (or silent)": (
            float(np.max(np.abs(audio_out))) >= 1.0 - 1e-5
            or float(np.max(np.abs(audio_out))) < 1e-6
        ),
        "No NaN values"                  : not np.any(np.isnan(audio_out)),
        "No Inf values"                  : not np.any(np.isinf(audio_out)),
    }

    all_pass = True
    for desc, result in checks.items():
        icon = "✓" if result else "✗"
        print(f"    {icon}  {desc}")
        if not result:
            all_pass = False

    # ── 5. Optionally save output ────────────────────────────────────────────
    if args.save:
        base, _ = os.path.splitext(wav_path)
        out_path = base + "_preprocessed.wav"
        try:
            save_wav(out_path, audio_out, out_sr)
            print(f"\n  Saved  → {out_path}")
        except Exception as exc:
            print(f"\n  WARNING: Could not save output WAV: {exc}", file=sys.stderr)

    # ── 6. Summary ───────────────────────────────────────────────────────────
    print(f"\n{SEP2}")
    if all_pass:
        print("  Result: ALL CHECKS PASSED ✓")
    else:
        print("  Result: SOME CHECKS FAILED ✗  (see above)")
        sys.exit(1)
    print(SEP2)


if __name__ == "__main__":
    main()
