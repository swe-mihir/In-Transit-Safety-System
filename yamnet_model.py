"""
yamnet_model.py — Background Sound Classifier (Model B in Fusion)
==================================================================
Uses Google's YAMNet (TensorFlow Hub) to classify environmental audio
and map the result to a 3-class safety distribution for fusion.py.

Public API
----------
  load_model()                   → (yamnet_model, class_names)
  detect_sound(audio_path, ...)  → (label: str, score: float)
  classify_sound(label, score)   → "NORMAL" | "RISKY" | "DANGER"
  get_fusion_output(audio_path, yamnet_model, class_names)
                                 → {"normal": float, "risky": float, "dangerous": float}
"""
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
CLASS_MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yamnet_class_map.csv")

# ---------------------------------------------------------------------------
# Lazy model cache
# ---------------------------------------------------------------------------

_yamnet_model = None
_class_names  = None


def load_model():
    """Load (and cache) YAMNet + class names.  Safe to call multiple times.

    Returns
    -------
    (yamnet_model, class_names: list[str])
    """
    global _yamnet_model, _class_names

    if _yamnet_model is not None:
        return _yamnet_model, _class_names

    import tensorflow_hub as hub

    print("[YAMNet] Loading model from TF Hub …")
    _yamnet_model = hub.load(YAMNET_URL)

    print("[YAMNet] Fetching class names …")
    _class_names = pd.read_csv(CLASS_MAP_PATH)["display_name"].tolist()

    print("[YAMNet] Ready.\n")
    return _yamnet_model, _class_names


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def detect_sound(audio_path: str, yamnet_model, class_names: list) -> tuple[str, float]:
    """Run YAMNet on a WAV file and return (top_label, mean_score).

    Parameters
    ----------
    audio_path   : path to a 16 kHz mono WAV file
    yamnet_model : loaded TF Hub YAMNet model
    class_names  : list of 521 class display names

    Returns
    -------
    (label: str, score: float)
    """
    import librosa

    audio, _ = librosa.load(audio_path, sr=16_000)
    scores, _embeddings, _spectrogram = yamnet_model(audio)
    mean_scores = np.mean(scores.numpy(), axis=0)
    top_index   = int(np.argmax(mean_scores))
    return class_names[top_index], float(mean_scores[top_index])


SAFE_KEYWORDS = ["speech", "music", "bird", "wind", "rain", "vehicle", "engine"]

def classify_sound(label: str, score: float) -> str:
    label_lc = label.lower()

    if any(w in label_lc for w in ["gunshot", "explosion", "glass", "crash"]):
        return "DANGER"
    elif any(w in label_lc for w in ["scream", "shout", "alarm", "cry"]):
        return "DANGER" if score >= 0.85 else "RISKY"
    elif any(w in label_lc for w in SAFE_KEYWORDS):
        return "NORMAL"
    else:
        # Unknown label — be conservative but not alarmist
        if score < 0.5:
            return "NORMAL"
        elif score < 0.85:
            return "RISKY"
        else:
            return "DANGER"


def get_fusion_output(
    audio_path: str,
    yamnet_model,
    class_names: list,
) -> dict:
    """Return the 3-key probability dict expected by fusion.fuse_outputs() as Model B.

    The hard category is converted to a peaked probability distribution so
    that the weighted-average fusion has numeric values to work with.

    Category → (normal, risky, dangerous)
    NORMAL   → (0.85,  0.10,  0.05)
    RISKY    → (0.10,  0.75,  0.15)
    DANGER   → (0.05,  0.10,  0.85)
    """
    label, score = detect_sound(audio_path, yamnet_model, class_names)
    category     = classify_sound(label, score)

    distribution = {
        "NORMAL": {"normal": 0.85, "risky": 0.10, "dangerous": 0.05},
        "RISKY":  {"normal": 0.10, "risky": 0.75, "dangerous": 0.15},
        "DANGER": {"normal": 0.05, "risky": 0.10, "dangerous": 0.85},
    }
    result = distribution[category].copy()
    result["_meta"] = {"label": label, "score": round(score, 4), "category": category}
    return result


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    audio_path = sys.argv[1] if len(sys.argv) > 1 else "test.wav"

    ym, cn = load_model()
    label, score = detect_sound(audio_path, ym, cn)
    level        = classify_sound(label, score)

    print(f"File     : {os.path.basename(audio_path)}")
    print(f"Label    : {label}")
    print(f"Score    : {score:.4f}")
    print(f"Category : {level}")

    fusion_out = get_fusion_output(audio_path, ym, cn)
    print(f"Fusion   : {fusion_out}")
