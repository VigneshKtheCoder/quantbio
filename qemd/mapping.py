import numpy as np
from .config import GAMMA_MIN, GAMMA_MAX, GAMMA_MID, EPS_ALPHA, J_BASE, J_BETA, GAMMA_LAM, J_MIN_CLIP, J_MAX_CLIP

def map_expression_to_epsilon(expr_vec_z, eps0=0.0):
    """
    Maps the mean z-scored gene expression (for subunits of a node) to the site energy (epsilon).
    epsilon = eps0 + alpha * z
    """
    # expr_vec_z: np.array of z-scored expression values for genes in this node
    z = expr_vec_z.mean()  # average z-score for that node’s genes
    epsilon = eps0 + EPS_ALPHA * z
    return float(epsilon)

def map_supercomplex_to_J(edge_expr_z):
    """
    Maps the mean z-scored expression of subunits involved in an edge to the coupling (J).
    J = J_base * (1 + beta * z)
    """
    # edge_expr_z: e.g. mean z of the two nodes / interacting subunits
    J = J_BASE * (1 + J_BETA * edge_expr_z)
    # Clip J to a realistic band (0.02–0.08)
    return float(np.clip(J, J_MIN_CLIP, J_MAX_CLIP))

def map_redox_to_gamma(redox_z):
    """
    Maps a z-score of a redox/hypoxia/ROS proxy to the decoherence rate (gamma).
    """
    # redox_z: z-score of some redox/hypoxia/ROS proxy per sample
    gamma = GAMMA_MID * (1 + GAMMA_LAM * redox_z)
    # Clip gamma to the defined sweep range
    return float(np.clip(gamma, GAMMA_MIN, GAMMA_MAX))

def get_params_for_sample(omics_data):
    """
    Placeholder function to simulate extracting physical parameters from omics data.
    In a real system, 'omics_data' would be a structured input (e.g., a Pandas row).
    """
    # Assume omics_data has keys for z-scores:
    # 'ci_z', 'ciii_z', 'civ_z' (for site energies)
    # 'ci_ciii_z', 'ciii_civ_z' (for couplings)
    # 'redox_z' (for gamma)

    # 1. Site Energies (ε) - simple 7-site chain example
    epsilon = [
        map_expression_to_epsilon(omics_data['ci_z']),
        map_expression_to_epsilon(omics_data['ci_z']),
        map_expression_to_epsilon(omics_data['ci_z']),
        map_expression_to_epsilon(omics_data['ciii_z']),
        map_expression_to_epsilon(omics_data['ciii_z']),
        map_expression_to_epsilon(omics_data['ciii_z']),
        map_expression_to_epsilon(omics_data['civ_z']),
    ]

    # 2. Couplings (J) - simplified nearest-neighbor chain
    J_base = map_supercomplex_to_J(omics_data['ci_ciii_z']) # Placeholder for complex-complex coupling
    J = [
        map_supercomplex_to_J(omics_data['ci_ciii_z']), # J_01
        map_supercomplex_to_J(omics_data['ci_ciii_z']), # J_12
        map_supercomplex_to_J(omics_data['ci_ciii_z']), # J_23 (CI-CIII interface)
        map_supercomplex_to_J(omics_data['ciii_civ_z']), # J_34
        map_supercomplex_to_J(omics_data['ciii_civ_z']), # J_45
        map_supercomplex_to_J(omics_data['ciii_civ_z']) # J_56 (CIII-CIV interface)
    ]

    # 3. Decoherence (γ)
    gamma = map_redox_to_gamma(omics_data['redox_z'])
    
    # 4. Sink/Loss are constants from config
    return {
        'epsilon': epsilon,
        'J': J,
        'gamma': gamma
    }
