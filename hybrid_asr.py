"""
hybrid_asr.py — Hybrid Hindi ASR Module
"""
import os
import wave
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
os.environ["PATH"] += os.pathsep + r"C:/ffmpeg-8.0.1-essentials_build/bin"
WAV2VEC_MODEL        = "./wav2vec2"
WHISPER_SIZE         = "small"
SAMPLE_RATE          = 16_000
CONFIDENCE_THRESHOLD = 0.4

# Temp file in same folder as this script — works on Windows
_TMP_WAV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_asr_tmp.wav")

# ---------------------------------------------------------------------------
# Lazy model cache
# ---------------------------------------------------------------------------

_whisper_model = None
_wav2vec       = None
_processor     = None


def load_models(
    wav2vec_path: str = WAV2VEC_MODEL,
    whisper_size: str = WHISPER_SIZE,
) -> tuple:
    global _whisper_model, _wav2vec, _processor

    if _whisper_model is not None:
        return _whisper_model, _wav2vec, _processor

    import whisper as _whisper

    gpu = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[ASR] Loading Whisper-{whisper_size} on {gpu} …")
    _whisper_model = _whisper.load_model(whisper_size, device=gpu)

    print("[ASR] Loading Wav2Vec2 on cpu …")
    wav2vec_abs = os.path.abspath(wav2vec_path).replace("\\", "/")
    _processor  = Wav2Vec2Processor.from_pretrained(wav2vec_abs, local_files_only=True)
    _wav2vec    = Wav2Vec2ForCTC.from_pretrained(wav2vec_abs, local_files_only=True).eval().to("cpu")

    print("[ASR] Both models ready.\n")
    return _whisper_model, _wav2vec, _processor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wav2vec_confidence(speech, wav2vec, processor) -> float:
    inputs = processor(speech, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = wav2vec(inputs.input_values.to("cpu")).logits
    probs = torch.softmax(logits, dim=-1)
    return round(probs.max(dim=-1).values.mean().item(), 4)


def _build_result(text: str, confidence: float) -> dict:
    return {
        "text":       text,
        "confidence": confidence,
        "quality":    "high" if confidence >= CONFIDENCE_THRESHOLD else "low",
        "flagged":    confidence < CONFIDENCE_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe(audio_path: str, whisper_model, wav2vec, processor) -> dict:
    import whisper as _whisper
    audio_path = os.path.abspath(audio_path).replace("\\", "/")  # ← ADD THIS
    speech = _whisper.load_audio(audio_path).astype(np.float32)
    result = whisper_model.transcribe(
        audio_path,
        language="hi",
        task="transcribe",
        fp16=torch.cuda.is_available(),
        verbose=False,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4,
        temperature=0.0,
    )
    text       = result["text"].strip()
    confidence = _wav2vec_confidence(speech, wav2vec, processor)
    return _build_result(text, confidence)


def transcribe_waveform(
    waveform: np.ndarray,
    whisper_model,
    wav2vec,
    processor,
    *,
    tmp_path: str = _TMP_WAV,          # ← Windows-safe path next to this file
) -> dict:
    """Transcribe directly from a float32 numpy waveform at 16 kHz."""
    pcm = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)

    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())

    try:
        result = transcribe(tmp_path, whisper_model, wav2vec, processor)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hybrid Hindi ASR")
    parser.add_argument("audio", help="Path to .wav file")
    args = parser.parse_args()

    wm, wv, proc = load_models()
    out = transcribe(args.audio, wm, wv, proc)
    print(f"Text       : {out['text']}")
    print(f"Confidence : {out['confidence']}")
    print(f"Quality    : {out['quality']}")
    print(f"Flagged    : {out['flagged']}")
