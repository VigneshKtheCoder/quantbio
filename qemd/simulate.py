import numpy as np
from scipy.linalg import expm
from .config import NUM_ETC_SITES, K_SINK, K_LOSS, SINK_INDEX, TIME_END, DT, GAMMAS_SWEEP
from .metrics import compute_ete_from_series, compute_tau_c

def build_hamiltonian(epsilon, J):
    """Build the Hamiltonian H for the N-site chain."""
    N = len(epsilon)
    H = np.diag(epsilon)
    
    # Add nearest-neighbor couplings
    for i in range(N - 1):
        H[i, i+1] = J[i]
        H[i+1, i] = J[i]
        
    return H

def build_lindblad_ops(num_sites, gamma, k_sink, k_loss):
    """
    Build the Lindblad superoperator components (jump operators L).
    L = [L_dephase_i] + [L_loss_i] + [L_sink]
    """
    L_ops = []
    
    # 1. Dephasing (Pure Decoherence)
    # L_i = sqrt(gamma) |i><i|
    for i in range(num_sites):
        A = np.zeros((num_sites, num_sites), dtype=complex)
        A[i, i] = 1.0
        L_ops.append(np.sqrt(gamma) * A)
        
    # 2. General Loss (Non-Productive Dissipation)
    # L_i = sqrt(k_loss) |i><i| for i != SINK_INDEX
    # This ensures density decays non-productively from all but the sink site
    for i in range(num_sites):
        if i != SINK_INDEX:
            A = np.zeros((num_sites, num_sites), dtype=complex)
            A[i, i] = 1.0
            L_ops.append(np.sqrt(k_loss) * A)

    # 3. Sink/Productive Dissipation (Out-Coupling)
    # L_sink = sqrt(k_sink) |SINK><SINK|
    A_sink = np.zeros((num_sites, num_sites), dtype=complex)
    A_sink[SINK_INDEX, SINK_INDEX] = 1.0
    L_ops.append(np.sqrt(k_sink) * A_sink)
    
    return L_ops

def time_evolve(rho0, H, L_ops, T_end, dt):
    """
    Time evolution of the density matrix (rho) using the Lindblad equation
    d(rho)/dt = -i [H, rho] + Sum_k (L_k* rho L_k - 0.5 {L_k* L_k, rho})
    Uses the superoperator formalism for numerical stability.
    """
    N = rho0.shape[0]
    
    # Superoperator for Hamiltonian part: L_H(rho) = -i [H, rho]
    # L_H * rho_vec = -i (I \otimes H - H^T \otimes I) * rho_vec
    L_H = -1j * (np.kron(np.identity(N), H) - np.kron(H.T, np.identity(N)))
    
    # Superoperator for Dissipative part: L_D(rho) = Sum_k L_k* rho L_k - 0.5 {L_k* L_k, rho}
    L_D = np.zeros((N*N, N*N), dtype=complex)
    for Lk in L_ops:
        Lk_dag = Lk.conj().T
        Lk_dag_Lk = Lk_dag @ Lk
        
        term1 = np.kron(Lk.conj(), Lk)
        term2 = 0.5 * (np.kron(np.identity(N), Lk_dag_Lk) + np.kron(Lk_dag_Lk.T, np.identity(N)))
        L_D += (term1 - term2)
        
    L_total = L_H + L_D
    
    # Time steps
    times = np.arange(0, T_end + dt, dt)
    rho_vec = rho0.flatten()
    rho_t_series = [rho0]
    
    # Evolution operator: exp(L_total * dt)
    U_dt = expm(L_total * dt)

    for _ in times[1:]:
        rho_vec = U_dt @ rho_vec
        rho_new = rho_vec.reshape((N, N))
        rho_t_series.append(rho_new)
        
    return rho_t_series, times

def compute_ete_for_gamma(params, gamma):
    """Run a single simulation for a given gamma and compute ETE."""
    epsilon, J = params['epsilon'], params['J']
    H = build_hamiltonian(NUM_ETC_SITES, epsilon, J)
    L_ops = build_lindblad_ops(NUM_ETC_SITES, gamma, K_SINK, K_LOSS)
    
    # Initial state: localized at site 0 (e.g., Complex I)
    rho0 = np.zeros((NUM_ETC_SITES, NUM_ETC_SITES), dtype=complex)
    rho0[0, 0] = 1.0 
    
    rho_t_series, _ = time_evolve(rho0, H, L_ops, TIME_END, DT)
    
    ete = compute_ete_from_series(rho_t_series, SINK_INDEX, K_SINK, DT)
    return ete

def enaqt_sweep(params):
    """
    Sweeps over the range of gamma values to find the ENAQT bell curve peak.
    """
    results = []
    for g in GAMMAS_SWEEP:
        ete = compute_ete_for_gamma(params, g)
        results.append({"gamma": g, "ETE": ete})
    return results

def run_full_simulation(params):
    """Runs the primary simulation using the omics-derived gamma."""
    epsilon, J, gamma = params['epsilon'], params['J'], params['gamma']
    H = build_hamiltonian(NUM_ETC_SITES, epsilon, J)
    L_ops = build_lindblad_ops(NUM_ETC_SITES, gamma, K_SINK, K_LOSS)
    
    # Initial state: localized at site 0
    rho0 = np.zeros((NUM_ETC_SITES, NUM_ETC_SITES), dtype=complex)
    rho0[0, 0] = 1.0 
    
    rho_t_series, times = time_evolve(rho0, H, L_ops, TIME_END, DT)

    # Compute all core metrics
    ete_instant = compute_ete_from_series(rho_t_series, SINK_INDEX, K_SINK, DT)
    tau_c = compute_tau_c(rho_t_series, times)
    
    return {
        "ETE_instant": ete_instant,
        "tau_c": tau_c,
        "rho_series": rho_t_series, # For post-hoc analysis/viz
        "times": times
    }
