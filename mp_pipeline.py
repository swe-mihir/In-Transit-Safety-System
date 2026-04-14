"""
pipeline_master.py — In-Transit Safety System Master Orchestrator
==================================================================

Full pipeline flow
------------------

  [server.py]                 — streams mic audio, saves WAV chunks
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
                    ↓
          [dashboard_server.py]         — live browser dashboard on :5050

Usage
-----
    python pipeline_master.py
    Then open http://localhost:5050 in your browser.
"""

import os
import queue
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------------------
# Sub-module imports
# ---------------------------------------------------------------------------

import server               as _server_mod
import audio_preprocessor   as _preproc
import hybrid_asr           as _asr
import model_b_inference    as _model_a     # sentiment  → "Model A" in fusion
import yamnet_model         as _yamnet      # sound FX   → "Model B" in fusion
import fusion               as _fusion
import dashboard_server     as _dashboard   # ← live browser dashboard

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_B_PATH = os.getenv("MODEL_B_PATH", "./sentiment")
WAV2VEC_PATH = os.getenv("WAV2VEC_PATH", "./wav2vec2")
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "small")

# ---------------------------------------------------------------------------
# Model handles
# ---------------------------------------------------------------------------

_whisper_model  = None
_wav2vec        = None
_asr_processor  = None
_bert_tokenizer = None
_bert_model     = None
_yamnet_model   = None
_yamnet_classes = None


def load_all_models():
    global _whisper_model, _wav2vec, _asr_processor
    global _bert_tokenizer, _bert_model
    global _yamnet_model, _yamnet_classes

    print("\n" + "=" * 60)
    print("  📦  Loading all models — please wait …")
    print("=" * 60)

    wav2vec_abs   = os.path.abspath(WAV2VEC_PATH).replace("\\", "/")
    sentiment_abs = os.path.abspath(MODEL_B_PATH).replace("\\", "/")

    _asr.WAV2VEC_MODEL = wav2vec_abs
    _asr.WHISPER_SIZE  = WHISPER_SIZE
    _whisper_model, _wav2vec, _asr_processor = _asr.load_models(
        wav2vec_path=wav2vec_abs,
        whisper_size=WHISPER_SIZE,
    )

    _model_a.MODEL_PATH = sentiment_abs
    _bert_tokenizer, _bert_model = _model_a.load_model(model_path=sentiment_abs)

    _yamnet_model, _yamnet_classes = _yamnet.load_model()

    print("=" * 60)
    print("  ✅  All models loaded.\n")


# ---------------------------------------------------------------------------
# Per-chunk pipeline
# ---------------------------------------------------------------------------

def _run_sentiment(hindi_text: str) -> dict:
    return _model_a.get_fusion_output(hindi_text, _bert_tokenizer, _bert_model)


def _run_yamnet(wav_path: str) -> dict:
    return _yamnet.get_fusion_output(wav_path, _yamnet_model, _yamnet_classes)


def process_chunk(wav_path: str) -> dict | None:
    t0 = time.time()
    print(f"\n{'─'*60}")
    print(f"[Master] Processing: {wav_path}")

    # Normalise path for Windows (ffmpeg needs forward slashes)
    wav_path_fwd = os.path.abspath(wav_path).replace("\\", "/")

    # ── Step 1: Preprocess ───────────────────────────────────────────────
    waveform, sr = _preproc.preprocess_from_file(wav_path_fwd)
    print(f"[Preprocess] done  ({len(waveform)/sr:.2f}s at {sr} Hz)")

    # ── Step 2: ASR ──────────────────────────────────────────────────────
    asr_result = _asr.transcribe(
        wav_path_fwd, _whisper_model, _wav2vec, _asr_processor
    )
    print(f"[ASR] text='{asr_result['text']}'  "
          f"confidence={asr_result['confidence']}  "
          f"quality={asr_result['quality']}")

    if asr_result["flagged"]:
        print("[ASR] ⚠️  Low-quality audio — skipping sentiment; YAMNet still runs.")

    hindi_text = asr_result["text"] if not asr_result["flagged"] else ""

    # ── Step 3: Parallel branches ─────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {}
        if hindi_text:
            futures["sentiment"] = pool.submit(_run_sentiment, hindi_text)
        futures["yamnet"] = pool.submit(_run_yamnet, wav_path_fwd)

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
    _neutral = {"normal": 0.34, "risky": 0.33, "dangerous": 0.33}
    if output_a is None and output_b is None:
        print("[Fusion] Both branches failed — cannot fuse.")
        return None
    if output_a is None:
        print("[Fusion] ⚠️  Sentiment unavailable — using neutral for Model A.")
        output_a = _neutral.copy()
    if output_b is None:
        print("[Fusion] ⚠️  YAMNet unavailable — using neutral for Model B.")
        output_b = _neutral.copy()

    fusion_result = _fusion.fuse_outputs(output_a, output_b)
    _fusion.print_result(fusion_result)

    elapsed = time.time() - t0
    print(f"[Master] ⏱️  Chunk processed in {elapsed:.2f}s")

    return fusion_result, asr_result, output_a, output_b


# ---------------------------------------------------------------------------
# Pipeline worker thread
# ---------------------------------------------------------------------------

def _pipeline_worker(stop_event: threading.Event):
    print("[Worker] Pipeline worker started — waiting for audio chunks …\n")

    while not stop_event.is_set():
        try:
            wav_path = _server_mod.audio_chunk_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        try:
            ret = process_chunk(wav_path)
            if ret:
                fusion_result, asr_result, output_a, output_b = ret
                _on_decision(wav_path, fusion_result, asr_result, output_a, output_b)
        except Exception as exc:
            print(f"[Worker] ❌  Unhandled error for {wav_path}: {exc}")
            traceback.print_exc()

    print("[Worker] Worker stopped.")


def _on_decision(wav_path: str, result: dict,
                 asr_result: dict = None,
                 output_a: dict = None,
                 output_b: dict = None):
    label = result["final_label"].upper()
    score = result["final_score"]

    # ── Console output ────────────────────────────────────────────────────
    print("\n" + "🔷" * 30)
    print(f"  FILE   : {os.path.basename(wav_path)}")
    print(f"  LABEL  : {label}")
    print(f"  SCORE  : {score}")
    print(f"  MODEL A (Sentiment) : {result['model_a_label'].upper()}")
    print(f"  MODEL B (YAMNet)    : {result['model_b_label'].upper()}")
    print(f"  ESCALATED           : {'YES ⚠️' if result['escalated'] else 'NO'}")
    if label == "DANGEROUS":
        print("\n  🚨  ALERT — DANGEROUS situation detected!")
    elif label == "RISKY":
        print("\n  ⚠️   WARNING — Risky situation detected.")
    else:
        print("\n  ✅  Situation is NORMAL.")
    print("🔷" * 30 + "\n")

    # ── Push to live dashboard ────────────────────────────────────────────
    try:
        _dashboard.push_decision(result, asr_result, output_a, output_b)
    except Exception as exc:
        print(f"[Dashboard] Push failed: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # 1. Start live dashboard (background thread, non-blocking)
    _dashboard.start_dashboard(port=5050)
    print("[Dashboard] Open http://localhost:5050 in your browser\n")

    # 2. Load all models
    load_all_models()

    # 3. Start pipeline worker
    stop_event = threading.Event()
    worker_thread = threading.Thread(
        target=_pipeline_worker,
        args=(stop_event,),
        daemon=True,
        name="PipelineWorker",
    )
    worker_thread.start()

    # 4. Start audio server (blocking)
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