# qemd/fusion.py
import numpy as np
from sklearn.preprocessing import minmax_scale

# Heuristic weights
W1_ETE = 1.0
W2_TAU_C = 1.0
W3_GAMMA_STAR = -1.0  # higher gamma* is bad
W4_RESILIENCE = 0.5
BIAS = 0.0

def _safe_minmax(values):
    """
    Min-max scale with safe behavior when all values are identical.
    Returns zeros if no variance.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    if np.allclose(values, values[0]):
        return np.zeros_like(values)
    return minmax_scale(values, feature_range=(0, 1))

def normalize_metrics(cohort_metrics):
    """
    Min-max normalize tau_c and gamma* across cohort.
    ETE_peak and resilience already constrained.
    """
    if not cohort_metrics:
        return []

    tau_c_values = np.array([m['tau_c'] for m in cohort_metrics], dtype=float)
    gamma_star_values = np.array([m['gamma_star'] for m in cohort_metrics], dtype=float)

    tau_c_norm = _safe_minmax(tau_c_values)
    gamma_star_norm = _safe_minmax(gamma_star_values)

    normalized = []
    for i, m in enumerate(cohort_metrics):
        normalized.append({
            "sample_id": m["sample_id"],
            "ETE_peak": float(np.clip(m["ETE_peak"], 0.0, 1.0)),
            "tau_c_norm": float(np.clip(tau_c_norm[i], 0.0, 1.0)),
            "gamma_star_norm": float(np.clip(gamma_star_norm[i], 0.0, 1.0)),
            "resilience": float(np.clip(m["resilience"], 0.0, 1.0)),
        })

    return normalized

def fuse_qhs(normalized_metrics):
    """
    Compute Quantum Health Score (QHS) via logistic fusion.
    Input is one sample's normalized metrics dict.
    """
    z = (
        W1_ETE * normalized_metrics["ETE_peak"] +
        W2_TAU_C * normalized_metrics["tau_c_norm"] +
        W3_GAMMA_STAR * normalized_metrics["gamma_star_norm"] +
        W4_RESILIENCE * normalized_metrics["resilience"] +
        BIAS
    )

    qhs = 1.0 / (1.0 + np.exp(-z))
    return float(qhs)

def compute_cohort_qhs(cohort_metrics):
    """
    End-to-end QHS for an entire cohort.
    Expects each entry to have:
      - sample_id
      - ETE_peak
      - tau_c
      - gamma_star
      - resilience
    """
    normalized_metrics = normalize_metrics(cohort_metrics)
    final_scores = []
    for m in normalized_metrics:
        qhs = fuse_qhs(m)
        final_scores.append({
            "sample_id": m["sample_id"],
            "QHS": qhs,
        })
    return final_scores
