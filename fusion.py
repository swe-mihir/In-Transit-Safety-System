"""
fusion.py — Output Fusion Module
=================================
Strategy : Sentiment-First with Both-Dangerous Escalation

Labels
------
  Normal    → score 0.0 – 0.3
  Risky     → score 0.3 – 0.7
  Dangerous → score 0.7 – 1.0

Public API
----------
  fuse_outputs(output_a, output_b)  → result dict
  get_label(score)                  → str
  print_result(result)              → None
"""

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "normal":    (0.0, 0.3),
    "risky":     (0.3, 0.7),
    "dangerous": (0.7, 1.0),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_label(score: float) -> str:
    """Map a numeric danger score to its safety label."""
    if score < 0.3:
        return "normal"
    elif score < 0.7:
        return "risky"
    else:
        return "dangerous"


def validate_output(model_output: dict, model_name: str) -> None:
    """Ensure required keys are present and values sum to ~1.0."""
    required_keys = {"normal", "risky", "dangerous"}
    if not required_keys.issubset(model_output.keys()):
        raise ValueError(f"{model_name} output missing keys. Expected: {required_keys}")
    total = sum(model_output[k] for k in required_keys)
    if not (0.99 <= total <= 1.01):
        raise ValueError(f"{model_name} output values must sum to ~1.0, got {total:.4f}")


def sentiment_score(output_a: dict) -> float:
    """Compute a single danger score from Model A (sentiment) using label midpoints."""
    midpoints = {"normal": 0.15, "risky": 0.50, "dangerous": 0.85}
    return round(sum(output_a[lbl] * midpoints[lbl] for lbl in midpoints), 4)


# ---------------------------------------------------------------------------
# Main fusion function
# ---------------------------------------------------------------------------

def fuse_outputs(output_a: dict, output_b: dict) -> dict:
    """Fuse Model A (sentiment) and Model B (YAMNet) outputs.

    Logic
    -----
    - Both dangerous → final label: dangerous
    - Otherwise      → pass through Model A (sentiment) label and score directly

    Parameters
    ----------
    output_a : {"normal": float, "risky": float, "dangerous": float}
    output_b : {"normal": float, "risky": float, "dangerous": float}

    Returns
    -------
    {
        "final_label":   str,
        "final_score":   float,
        "escalated":     bool,
        "model_a_label": str,
        "model_b_label": str,
    }
    """
    # Strip any _meta keys before validation
    clean_a = {k: v for k, v in output_a.items() if k in {"normal", "risky", "dangerous"}}
    clean_b = {k: v for k, v in output_b.items() if k in {"normal", "risky", "dangerous"}}

    validate_output(clean_a, "Model A")
    validate_output(clean_b, "Model B")

    label_a = max(clean_a, key=clean_a.get)
    label_b = max(clean_b, key=clean_b.get)
    score_a = sentiment_score(clean_a)

    both_dangerous = (label_a == "dangerous") and (label_b == "dangerous")

    if both_dangerous:
        return {
            "final_label":   "dangerous",
            "final_score":   score_a,
            "escalated":     True,
            "model_a_label": label_a,
            "model_b_label": label_b,
        }

    return {
        "final_label":   label_a,
        "final_score":   score_a,
        "escalated":     False,
        "model_a_label": label_a,
        "model_b_label": label_b,
    }


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_result(result: dict) -> None:
    print("\n" + "=" * 45)
    print("       IN-TRANSIT SAFETY — FUSION RESULT")
    print("=" * 45)
    print(f"  Model A (Sentiment) : {result['model_a_label'].upper()}")
    print(f"  Model B (YAMNet)    : {result['model_b_label'].upper()}")
    print(f"  Safety Escalated    : {'YES ⚠️' if result['escalated'] else 'NO'}")
    print(f"  ► Final Label       : {result['final_label'].upper()}")
    print(f"  ► Final Score       : {result['final_score']}")
    print("=" * 45 + "\n")