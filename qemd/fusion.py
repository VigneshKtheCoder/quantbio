import numpy as np
from sklearn.preprocessing import minmax_scale

# --- Static Fusion Parameters (for a first pass without training) ---
# These weights are heuristic: Higher ETE, tau_c, Resilience = GOOD
# Higher gamma* (more decoherence needed) = BAD (implies structural defect)
W1_ETE = 1.0
W2_TAU_C = 1.0
W3_GAMMA_STAR = -1.0 # Note the negative sign: high gamma* reduces score
W4_RESILIENCE = 0.5
BIAS = 0.0 # Bias term

def normalize_metrics(cohort_metrics):
    """
    Performs Min-Max normalization for tau_c and gamma* across the entire cohort.
    ETE_peak and Resilience are already in [0,1].
    """
    if not cohort_metrics:
        return []
        
    # Extract arrays
    tau_c_values = np.array([m['tau_c'] for m in cohort_metrics])
    gamma_star_values = np.array([m['gamma_star'] for m in cohort_metrics])

    # Normalize
    # Clip to [0,1] using minmax_scale
    tau_c_norm = minmax_scale(tau_c_values, feature_range=(0, 1))
    gamma_star_norm = minmax_scale(gamma_star_values, feature_range=(0, 1))
    
    # Re-package the results
    normalized_metrics = []
    for i, m in enumerate(cohort_metrics):
        normalized_metrics.append({
            'sample_id': m['sample_id'],
            'ETE_peak': m['ETE_peak'],
            'tau_c_norm': tau_c_norm[i],
            'gamma_star_norm': gamma_star_norm[i],
            'resilience': m['resilience']
        })
        
    return normalized_metrics

def fuse_qhs(normalized_metrics):
    """
    Fuses the normalized physical metrics into a single Quantum Health Score (QHS)
    using a logistic (sigmoid) function. This ensures the final score is in [0,1].
    QHS = 1 / (1 + exp(-z)), where z is a weighted sum of metrics.
    """
    # Create the weighted sum (z-score for the sigmoid)
    z = (W1_ETE * normalized_metrics['ETE_peak'] +
         W2_TAU_C * normalized_metrics['tau_c_norm'] +
         W3_GAMMA_STAR * normalized_metrics['gamma_star_norm'] +
         W4_RESILIENCE * normalized_metrics['resilience'] +
         BIAS)
         
    # Apply the sigmoid function to map z to [0, 1]
    qhs = 1.0 / (1.0 + np.exp(-z))
    return float(qhs)

def compute_cohort_qhs(cohort_metrics):
    """
    End-to-end function for computing QHS for an entire cohort.
    """
    normalized_metrics = normalize_metrics(cohort_metrics)
    
    final_scores = []
    for m in normalized_metrics:
        qhs = fuse_qhs(m)
        final_scores.append({
            'sample_id': m['sample_id'],
            'QHS': qhs
        })
        
    return final_scores
