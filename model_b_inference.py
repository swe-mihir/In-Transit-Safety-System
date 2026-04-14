"""
model_b_inference.py — Hindi Sentiment Model (Model A in Fusion)
=================================================================
Architecture : BertForSequenceClassification
Language     : Hindi text (Devanagari)
Labels       : 0 → normal | 1 → risky | 2 → danger

Public API
----------
  load_model(path)                        → (tokenizer, model)
  predict(text, tokenizer, model)         → full result dict
  get_fusion_output(text, tokenizer, model) → {"normal", "risky", "dangerous"}
"""
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BertForSequenceClassification

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = "./model_b"        # folder with config.json, model.safetensors, tokenizer.*

LABEL_MAP = {0: "normal", 1: "risky", 2: "danger"}

# ---------------------------------------------------------------------------
# Lazy model cache
# ---------------------------------------------------------------------------

_tokenizer = None
_model     = None


def load_model(model_path: str = MODEL_PATH):
    """Load (and cache) the BERT classifier.  Safe to call multiple times.

    Returns
    -------
    (tokenizer, model)
    """
    global _tokenizer, _model

    if _tokenizer is not None:
        return _tokenizer, _model

    print(f"[Model B] Loading from: {model_path}")
    model_abs = os.path.abspath(model_path)
    _tokenizer = AutoTokenizer.from_pretrained(model_abs)
    _model = BertForSequenceClassification.from_pretrained(
        model_abs, local_files_only=True
    ).eval()
    print("[Model B] Loaded ✓\n")
    return _tokenizer, _model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(hindi_text: str, tokenizer, model) -> dict:
    """Run inference on Hindi text.

    Returns
    -------
    dict with keys: normal, risky, danger, predicted_label, predicted_score
    """
    if not hindi_text or not hindi_text.strip():
        raise ValueError("Input text is empty.")

    inputs = tokenizer(
        hindi_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits          # (1, 3)

    probs = F.softmax(logits, dim=-1).squeeze()  # (3,)

    normal_prob = round(probs[0].item(), 4)
    risky_prob  = round(probs[1].item(), 4)
    danger_prob = round(probs[2].item(), 4)

    pred_idx   = probs.argmax().item()
    pred_label = LABEL_MAP[pred_idx]
    pred_score = round(probs[pred_idx].item(), 4)

    return {
        "normal":          normal_prob,
        "risky":           risky_prob,
        "danger":          danger_prob,
        "predicted_label": pred_label,
        "predicted_score": pred_score,
    }


def get_fusion_output(hindi_text: str, tokenizer, model) -> dict:
    """Return the 3-key dict expected by fusion.fuse_outputs() as Model A input.

    Keys: normal, risky, dangerous  (note: 'danger' → 'dangerous' rename).
    """
    result = predict(hindi_text, tokenizer, model)
    return {
        "normal":    result["normal"],
        "risky":     result["risky"],
        "dangerous": result["danger"],
    }


# ---------------------------------------------------------------------------
# Pretty print helper
# ---------------------------------------------------------------------------

def print_result(text: str, result: dict) -> None:
    print("\n" + "=" * 45)
    print("        MODEL B — HINDI SENTIMENT RESULT")
    print("=" * 45)
    print(f"  Input  : {text}")
    print(f"  Normal : {result['normal']:.4f}")
    print(f"  Risky  : {result['risky']:.4f}")
    print(f"  Danger : {result['danger']:.4f}")
    print(f"  ► Predicted : {result['predicted_label'].upper()}")
    print(f"  ► Score     : {result['predicted_score']:.4f}")
    print("=" * 45 + "\n")


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tok, mdl = load_model()
    test_inputs = [
        "आज मौसम बहुत अच्छा है।",
        "तुम्हें इसकी कीमत चुकानी होगी।",
        "मैं तुम्हें जान से मार दूंगा।",
    ]
    for text in test_inputs:
        result = predict(text, tok, mdl)
        print_result(text, result)
