import numpy as np

def compute_ete_from_series(rho_t_series, sink_index, k_sink, dt):
    """
    Computes Energy Transfer Efficiency (ETE) as the integral of the probability 
    current into the sink over the entire simulation time.
    """
    ete = 0.0
    for rho in rho_t_series:
        # P_sink = population at the sink site
        sink_pop = np.real(rho[sink_index, sink_index])
        # Current = k_sink * P_sink
        ete += k_sink * sink_pop * dt
        
    return float(np.clip(ete, 0.0, 1.0)) # Clamp to [0,1] for efficiency

def coherence_measure(rho):
    """
    Quantifies the total coherence (off-diagonal elements) of the density matrix.
    Uses the Frobenius norm of the off-diagonal part: ||rho - diag(rho)||_F.
    """
    off_diag = rho.copy()
    np.fill_diagonal(off_diag, 0)
    return float(np.linalg.norm(off_diag, ord='fro'))

def compute_tau_c(rho_t_series, times):
    """
    Computes the coherence lifetime (tau_c) as the time-average of the coherence measure.
    tau_c = Integral(t * C(t) dt) / Integral(C(t) dt)
    """
    C = np.array([coherence_measure(rho) for rho in rho_t_series])
    
    # Denominator: total area under the coherence curve
    den = np.trapz(C, times)
    
    if den < 1e-12:
        return 0.0 # No coherence observed, lifetime is zero
        
    # Numerator: moment of the coherence curve
    num = np.trapz(times * C, times)
    
    tau = num / den
    
    # Convert result (in ps) to a reasonable range by clipping
    # Normalization across a cohort will happen later (in fusion.py)
    return float(tau)

def find_gamma_star_and_ete_peak(enaqt_results):
    """
    Finds the optimal decoherence rate (gamma*) and the corresponding 
    peak ETE (ETE_peak) from the ENAQT sweep results.
    """
    if not enaqt_results:
        return 0.0, 0.0
        
    # Find the result dict with the maximum ETE
    best = max(enaqt_results, key=lambda d: d["ETE"])
    return best["gamma"], best["ETE"]

def compute_resilience(base_qhs, jitter_qhs_list):
    """
    Computes Resilience (R) based on the mean drop in QHS under parameter jitter.
    R = 1 - Mean(Drop Percentage)
    """
    drops = []
    for q_jitter in jitter_qhs_list:
        # Drop is the relative loss compared to the base QHS
        drop = max(0.0, (base_qhs - q_jitter) / max(base_qhs, 1e-6))
        drops.append(drop)
        
    mean_drop = np.mean(drops) if drops else 0.0
    R = 1.0 - mean_drop # R=1 is robust, R=0 is fragile
    return float(np.clip(R, 0.0, 1.0))
