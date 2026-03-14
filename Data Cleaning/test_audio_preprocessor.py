"""
test_audio_preprocessor.py
---------------------------
Tests the audio_preprocessor module with six scenarios covering every
code-path in preprocess_audio_chunk.

Test 1  – Mono 1-D array at 16 kHz          (pass-through)
Test 2  – Stereo (2, N) at 16 kHz           (channels-first)
Test 3  – Stereo (N, 2) at 44.1 kHz         (channels-last + resample)
Test 4  – Stereo (N, 2) at 48 kHz           (channels-last + resample)
Test 5  – Silent mono chunk                  (zero-division guard)
Test 6  – Streaming simulation (50 ms chunks, stereo 44.1 kHz)

Run:
    python test_audio_preprocessor.py
"""

from __future__ import annotations

import sys
import os

import numpy as np

# Allow running from the outputs directory without installing the package
sys.path.insert(0, os.path.dirname(__file__))
from audio_preprocessor import preprocess_audio_chunk, preprocess_stream, TARGET_SR

SEP = "─" * 64


# ---------------------------------------------------------------------------
# Shared assertion helpers
# ---------------------------------------------------------------------------

def assert_mono(w: np.ndarray, label: str) -> None:
    assert w.ndim == 1, f"[{label}] Expected 1-D, got shape {w.shape}"


def assert_normalised(w: np.ndarray, label: str, tol: float = 1e-5) -> None:
    peak = float(np.max(np.abs(w)))
    assert peak <= 1.0 + tol, f"[{label}] Peak {peak:.6f} > 1.0"
    if peak > 1e-6:
        assert peak >= 1.0 - tol, (
            f"[{label}] Non-silent chunk peak {peak:.6f} not close to 1.0"
        )


def assert_sr(out_sr: int, expected: int, label: str) -> None:
    assert out_sr == expected, (
        f"[{label}] Expected out_sr={expected}, got {out_sr}"
    )


def print_stats(label: str, w: np.ndarray, out_sr: int, in_channels: int,
                in_sr: int) -> None:
    print(f"\n  {label}")
    print(f"    Input  : {in_channels}-channel, {in_sr:,} Hz")
    print(f"    Output : {w.ndim}-D, shape={w.shape}, dtype={w.dtype}")
    print(f"    Samples: {len(w):,}   Duration: {len(w)/out_sr*1000:.1f} ms")
    print(f"    Max |x|: {np.max(np.abs(w)):.6f}   "
          f"Min: {w.min():.6f}   Max: {w.max():.6f}   "
          f"RMS: {np.sqrt(np.mean(w**2)):.6f}")


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------

def test_mono_16k() -> None:
    sr = TARGET_SR
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    chunk = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    out, out_sr = preprocess_audio_chunk(chunk, sr)

    assert_mono(out, "test_mono_16k")
    assert_normalised(out, "test_mono_16k")
    assert_sr(out_sr, TARGET_SR, "test_mono_16k")
    print_stats("Test 1 – Mono 16 kHz (A4 440 Hz sine)", out, out_sr, 1, sr)
    print("    ✓  PASSED")


def test_stereo_channels_first_16k() -> None:
    sr = TARGET_SR
    n = int(sr * 0.5)
    t = np.linspace(0, 0.5, n, endpoint=False)
    left  = (0.6 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    right = (0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    chunk = np.stack([left, right], axis=0)         # (2, N)

    out, out_sr = preprocess_audio_chunk(chunk, sr)

    assert_mono(out, "test_stereo_cf_16k")
    assert_normalised(out, "test_stereo_cf_16k")
    assert_sr(out_sr, TARGET_SR, "test_stereo_cf_16k")
    print_stats("Test 2 – Stereo (2,N) 16 kHz", out, out_sr, 2, sr)
    print("    ✓  PASSED")


def test_stereo_channels_last_44k() -> None:
    sr = 44_100
    n  = int(sr * 0.5)
    t  = np.linspace(0, 0.5, n, endpoint=False)
    left  = (0.8 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    right = (0.5 * np.cos(2 * np.pi * 600 * t)).astype(np.float32)
    chunk = np.stack([left, right], axis=1)         # (N, 2)

    out, out_sr = preprocess_audio_chunk(chunk, sr)

    assert_mono(out, "test_stereo_cl_44k")
    assert_normalised(out, "test_stereo_cl_44k")
    assert_sr(out_sr, TARGET_SR, "test_stereo_cl_44k")
    # Length check: 44100→16000, 0.5 s => ~8000 samples
    expected = int(round(n * TARGET_SR / sr))
    assert abs(len(out) - expected) <= 4, (
        f"Resampled length {len(out)} far from expected {expected}"
    )
    print_stats("Test 3 – Stereo (N,2) 44.1 kHz → 16 kHz", out, out_sr, 2, sr)
    print("    ✓  PASSED")


def test_stereo_channels_last_48k() -> None:
    sr = 48_000
    n  = int(sr * 0.5)
    t  = np.linspace(0, 0.5, n, endpoint=False)
    left  = (0.7 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    right = (0.7 * np.cos(2 * np.pi * 1000 * t)).astype(np.float32)
    chunk = np.stack([left, right], axis=1)         # (N, 2)

    out, out_sr = preprocess_audio_chunk(chunk, sr)

    assert_mono(out, "test_stereo_48k")
    assert_normalised(out, "test_stereo_48k")
    assert_sr(out_sr, TARGET_SR, "test_stereo_48k")
    expected = int(round(n * TARGET_SR / sr))
    assert abs(len(out) - expected) <= 4, (
        f"Resampled length {len(out)} far from expected {expected}"
    )
    print_stats("Test 4 – Stereo (N,2) 48 kHz → 16 kHz", out, out_sr, 2, sr)
    print("    ✓  PASSED")


def test_silent_chunk() -> None:
    sr    = TARGET_SR
    chunk = np.zeros(sr // 4, dtype=np.float32)     # 250 ms silence

    out, out_sr = preprocess_audio_chunk(chunk, sr)

    assert_mono(out, "test_silent")
    assert np.all(out == 0.0), "Silent chunk must remain all-zero"
    assert_sr(out_sr, TARGET_SR, "test_silent")
    print_stats("Test 5 – Silent mono chunk (all zeros)", out, out_sr, 1, sr)
    print("    ✓  PASSED")


def test_streaming_simulation() -> None:
    """Simulate a 3-second stereo stream at 44.1 kHz in 50 ms chunks."""
    total_dur  = 3.0
    sr         = 44_100
    chunk_dur  = 0.05                               # 50 ms per buffer
    chunk_n    = int(sr * chunk_dur)
    total_n    = int(sr * total_dur)
    rng        = np.random.default_rng(42)

    t      = np.linspace(0, total_dur, total_n, endpoint=False)
    speech = np.sin(2 * np.pi * 250 * t).astype(np.float32)
    noise  = (0.05 * rng.standard_normal(total_n)).astype(np.float32)

    # Build stereo stream with different levels per channel
    full_L = speech + noise
    full_R = 0.7 * speech + noise

    chunks: list[np.ndarray] = []
    for start in range(0, total_n, chunk_n):
        end = start + chunk_n
        L   = full_L[start:end]
        R   = full_R[start:end]
        if len(L) < chunk_n:           # zero-pad final short chunk
            pad = chunk_n - len(L)
            L   = np.pad(L, (0, pad))
            R   = np.pad(R, (0, pad))
        chunks.append(np.stack([L, R], axis=0))    # (2, chunk_n)

    processed = preprocess_stream(chunks, sr)

    all_pass = True
    for i, w in enumerate(processed):
        try:
            assert_mono(w, f"stream_chunk_{i}")
            assert_normalised(w, f"stream_chunk_{i}")
        except AssertionError as exc:
            print(f"    ✗ Chunk {i} FAILED: {exc}")
            all_pass = False

    reconstructed = np.concatenate(processed)
    print(f"\n  Test 6 – Streaming simulation (3 s, stereo 44.1 kHz, 50 ms chunks)")
    print(f"    Chunks processed       : {len(processed)}")
    print(f"    Chunk duration         : {chunk_dur*1000:.0f} ms  "
          f"({chunk_n} samples in, "
          f"{int(chunk_n * TARGET_SR / sr)} samples out)")
    print(f"    Reconstructed length   : {len(reconstructed):,} samples")
    print(f"    All peaks ≤ 1.0        : "
          f"{all(np.max(np.abs(c)) <= 1.0 + 1e-5 for c in processed)}")
    if all_pass:
        print("    ✓  PASSED")
    else:
        print("    ✗  FAILED (see above)")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(SEP)
    print("  Audio Preprocessor — Unit Tests")
    print(SEP)

    test_mono_16k()
    test_stereo_channels_first_16k()
    test_stereo_channels_last_44k()
    test_stereo_channels_last_48k()
    test_silent_chunk()
    test_streaming_simulation()

    print(f"\n{SEP}")
    print("  All 6 tests completed successfully. ✓")
    print(SEP)
