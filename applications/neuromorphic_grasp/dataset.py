import os
import sys
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def generate_grasp_scenario(total_time=10.0, dt=0.001):
    """
    Generates a high-dynamic object slippage and grasp force scenario.
    Returns time array and external shear load force trajectory (N).
    - t=0.0 to 3.0s: Static load (no external shear force).
    - t=3.0 to 7.0s: Object is pulled with a shear force of 4.5 N.
    - t=7.0 to 10.0s: The pull force increases to 6.0 N.
    """
    steps = int(total_time / dt)
    t = np.linspace(0, total_time, steps)
    F_ext = np.zeros_like(t)
    
    F_ext[t >= 3.0] = 4.5
    F_ext[t >= 7.0] = 6.0
    
    return t, F_ext
