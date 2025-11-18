import numpy as np

# --- Simulation Constants ---
NUM_ETC_SITES = 7 # Example: simplified 7-node ETC model
TIME_END = 50.0 # ps (Picoseconds - total evolution time)
DT = 0.05       # ps (Time step for Lindblad evolution)

# --- Decoherence (Gamma) Sweep Range for ENAQT Curve ---
# These define the range of the environment-induced decoherence rate (gamma)
GAMMA_MIN = 0.005 # /ps
GAMMA_MAX = 0.05  # /ps
GAMMA_SWEEP_STEPS = 20
GAMMAS_SWEEP = np.linspace(GAMMA_MIN, GAMMA_MAX, GAMMA_SWEEP_STEPS)
GAMMA_MID = 0.02

# --- Dissipation Constants ---
K_SINK = 0.03 # /ps (Rate of transition out of the system at the final site, e.g., site 6)
K_LOSS = 0.005 # /ps (Non-productive, non-coherent loss rate from all sites)
SINK_INDEX = NUM_ETC_SITES - 1 # The last site (e.g., Complex IV)

# --- Mapping Hyperparameters ---
EPS_ALPHA = 0.05  # Scale for z-score to site energy perturbation (ε)
J_BASE = 0.05     # Base coupling strength for all connected sites (J)
J_BETA = 0.3      # Scale for z-score to coupling perturbation (J)
GAMMA_LAM = 0.4   # Scale for redox z-score to decoherence rate (γ)
J_MIN_CLIP = 0.02
J_MAX_CLIP = 0.08
