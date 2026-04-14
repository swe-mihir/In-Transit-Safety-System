"""
pipeline_master.py — In-Transit Safety System Master Orchestrator
==================================================================

Full pipeline flow
------------------

  [server.py]                 — streams mic audio, saves 10-s WAV chunks
       ↓  (queue)
  [audio_preprocessor.py]    — mono, resample to 16 kHz, peak-normalise
       ↓
  [hybrid_asr.py]             — Whisper transcription + Wav2Vec2 quality gate
       ↓  (Hindi text)
  ┌────────────────┐  ┌─────────────────┐   ← run in parallel threads
  │ model_b_       │  │ yamnet_model.py │
  │ inference.py   │  │ (background SFX)│
  │ (Model A)      │  │ (Model B)       │
  └──────┬─────────┘  └──────┬──────────┘
         └──────────┬─────────┘
               [fusion.py]              — weighted average + safety-first
                    ↓
              Final Decision             NORMAL / RISKY / DANGEROUS

Usage
-----
    python pipeline_master.py

Environment variables (optional overrides)
------------------------------------------
    MODEL_B_PATH   path to BERT model folder   (default: ./model_b)
    WAV2VEC_PATH   path to Wav2Vec2 folder      (default: ./wav2vec2_model)
    WHISPER_SIZE   tiny|base|small|medium|large  (default: small)
"""

import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Sub-module imports
# ---------------------------------------------------------------------------

import server                   as _server_mod
import audio_preprocessor       as _preproc
import hybrid_asr               as _asr
import model_b_inference        as _model_a     # sentiment  → "Model A" in fusion
import yamnet_model             as _yamnet      # sound FX   → "Model B" in fusion
import fusion                   as _fusion

# ---------------------------------------------------------------------------
# Configuration (can be overridden via env vars)
# ---------------------------------------------------------------------------

MODEL_B_PATH  = os.getenv("MODEL_B_PATH",  "./sentiment")
WAV2VEC_PATH  = os.getenv("WAV2VEC_PATH",  "./wav2vec2")
WHISPER_SIZE  = os.getenv("WHISPER_SIZE",  "small")

# ---------------------------------------------------------------------------
# Module-level model handles (loaded once at startup)
# ---------------------------------------------------------------------------

_whisper_model = None
_wav2vec       = None
_asr_processor = None
_bert_tokenizer = None
_bert_model     = None
_yamnet_model   = None
_yamnet_classes = None


def load_all_models():
    """Load all four models. Called once before the pipeline loop starts."""
    global _whisper_model, _wav2vec, _asr_processor
    global _bert_tokenizer, _bert_model
    global _yamnet_model, _yamnet_classes

    print("\n" + "=" * 60)
    print("  📦  Loading all models — please wait …")
    print("=" * 60)

    # Resolve all paths to absolute, forward-slash form (fixes Windows HF bug)
    wav2vec_abs  = os.path.abspath(WAV2VEC_PATH).replace("\\", "/")
    sentiment_abs = os.path.abspath(MODEL_B_PATH).replace("\\", "/")

    # ASR models (Whisper + Wav2Vec2)
    _asr.WAV2VEC_MODEL = wav2vec_abs
    _asr.WHISPER_SIZE  = WHISPER_SIZE
    _whisper_model, _wav2vec, _asr_processor = _asr.load_models(
        wav2vec_path=wav2vec_abs,
        whisper_size=WHISPER_SIZE,
    )

    # Sentiment / BERT (Model A)
    _model_a.MODEL_PATH = sentiment_abs
    _bert_tokenizer, _bert_model = _model_a.load_model(model_path=sentiment_abs)

    # YAMNet (Model B)
    _yamnet_model, _yamnet_classes = _yamnet.load_model()

    print("=" * 60)
    print("  ✅  All models loaded.\n")


# ---------------------------------------------------------------------------
# Per-chunk pipeline
# ---------------------------------------------------------------------------

def _run_sentiment(hindi_text: str) -> dict:
    """Branch A: Hindi text → sentiment probabilities (fusion Model A)."""
    return _model_a.get_fusion_output(hindi_text, _bert_tokenizer, _bert_model)


def _run_yamnet(wav_path: str) -> dict:
    """Branch B: raw WAV → background-sound probabilities (fusion Model B)."""
    return _yamnet.get_fusion_output(wav_path, _yamnet_model, _yamnet_classes)


def process_chunk(wav_path: str) -> dict | None:
    """
    Full pipeline for one WAV chunk.

    Returns
    -------
    dict  — fusion result, or None if chunk was flagged as low-quality audio.
    """
    t0 = time.time()
    print(f"\n{'─'*60}")
    print(f"[Master] Processing: {wav_path}")

    # ── Step 1: Preprocess ───────────────────────────────────────────────
    waveform, sr = _preproc.preprocess_from_file(wav_path)
    print(f"[Preprocess] done  ({len(waveform)/sr:.2f}s at {sr} Hz)")

    # ── Step 2: ASR (Whisper + Wav2Vec2 quality gate) ────────────────────
    asr_result = _asr.transcribe(
        wav_path.replace("\\", "/"), _whisper_model, _wav2vec, _asr_processor
    )
    print(f"[ASR] text='{asr_result['text']}'  "
          f"confidence={asr_result['confidence']}  "
          f"quality={asr_result['quality']}")

    if asr_result["flagged"]:
        print("[ASR] ⚠️  Low-quality audio — skipping sentiment; YAMNet still runs.")

    hindi_text = asr_result["text"] if not asr_result["flagged"] else ""

    # ── Step 3: Parallel branches ─────────────────────────────────────────
    #   • Branch A — Sentiment (Model A) — needs text
    #   • Branch B — YAMNet   (Model B) — needs wav path
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {}

        if hindi_text:
            futures["sentiment"] = pool.submit(_run_sentiment, hindi_text)
        futures["yamnet"] = pool.submit(_run_yamnet, wav_path.replace("\\", "/"))

        results = {}
        for key, future in futures.items():
            try:
                results[key] = future.result(timeout=60)
            except Exception as exc:
                print(f"[{key}] ❌ Error: {exc}")
                results[key] = None

    output_a = results.get("sentiment")
    output_b = results.get("yamnet")

    # ── Step 4: Fusion ────────────────────────────────────────────────────
    if output_a is None and output_b is None:
        print("[Fusion] Both branches failed — cannot fuse.")
        return None

    # If one branch failed, use a neutral distribution for the missing one
    _neutral = {"normal": 0.34, "risky": 0.33, "dangerous": 0.33}
    if output_a is None:
        print("[Fusion] ⚠️  Sentiment unavailable — using neutral distribution for Model A.")
        output_a = _neutral.copy()
    if output_b is None:
        print("[Fusion] ⚠️  YAMNet unavailable — using neutral distribution for Model B.")
        output_b = _neutral.copy()

    fusion_result = _fusion.fuse_outputs(output_a, output_b)
    _fusion.print_result(fusion_result)

    elapsed = time.time() - t0
    print(f"[Master] ⏱️  Chunk processed in {elapsed:.2f}s")

    return fusion_result


# ---------------------------------------------------------------------------
# Pipeline worker thread
# ---------------------------------------------------------------------------

def _pipeline_worker(stop_event: threading.Event):
    """Continuously drain audio_chunk_queue and process each WAV file."""
    print("[Worker] Pipeline worker started — waiting for audio chunks …\n")

    while not stop_event.is_set():
        try:
            wav_path = _server_mod.audio_chunk_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        try:
            result = process_chunk(wav_path)
            if result:
                _on_decision(wav_path, result)
        except Exception as exc:
            import traceback
            print(f"[Worker] ❌  Unhandled error for {wav_path}: {exc}")
            traceback.print_exc()
    print("[Worker] Worker stopped.")


def _on_decision(wav_path: str, result: dict):
    """
    Called after every successful fusion decision.
    Extend this function to trigger alerts, log to a database, send to a UI, etc.
    """
    label = result["final_label"].upper()
    score = result["final_score"]

    if label == "DANGEROUS":
        print(f"\n🚨  ALERT — DANGEROUS situation detected! (score={score})")
        # TODO: trigger siren, send push notification, call emergency API, etc.
    elif label == "RISKY":
        print(f"\n⚠️   WARNING — Risky situation detected. (score={score})")
        # TODO: notify driver / operator
    else:
        print(f"\n✅  Situation is NORMAL. (score={score})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # 1. Load all models (blocking, runs before the server starts)
    load_all_models()

    # 2. Start the pipeline worker in a background thread
    stop_event = threading.Event()
    worker_thread = threading.Thread(
        target=_pipeline_worker,
        args=(stop_event,),
        daemon=True,
        name="PipelineWorker",
    )
    worker_thread.start()

    # 3. Start the streaming server (blocking — runs until Ctrl-C)
    try:
        _server_mod.start_server()
    except (KeyboardInterrupt, SystemExit):
        print("\n[Master] Shutting down …")
    finally:
        stop_event.set()
        worker_thread.join(timeout=5)
        print("[Master] Goodbye.")


if __name__ == "__main__":
    main()
