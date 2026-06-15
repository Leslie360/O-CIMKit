import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def generate_reference_trajectory(total_time=10.0, dt=0.001):
    """
    Generates a high-dynamic reference speed trajectory (target speed profiles over time).
    Consists of step-functions jumping to different target speeds.
    """
    steps = int(total_time / dt)
    t = np.linspace(0, total_time, steps)
    r = np.zeros_like(t)
    
    # 0.0s to 2.5s: target speed = 10.0 rad/s
    # 2.5s to 5.0s: target speed = 25.0 rad/s
    # 5.0s to 7.5s: target speed = 15.0 rad/s
    # 7.5s to 10.0s: target speed = 5.0 rad/s
    r[(t >= 0.0) & (t < 2.5)] = 10.0
    r[(t >= 2.5) & (t < 5.0)] = 25.0
    r[(t >= 5.0) & (t < 7.5)] = 15.0
    r[(t >= 7.5)] = 5.0
    
    return t, r
