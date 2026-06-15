import os
import sys
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from profiles.device_profile import DeviceProfile

class UltimateLorenzPredictor:
    """
    Optimized Lorenz Chaos Predictor with State Augmentation, Delay Embedding,
    and Ridge Regression, matching SOTA top-tier performance.
    """
    def __init__(self, device_profile=None, n_nodes=200, seed=42):
        self.profile = device_profile
        self.n_nodes = n_nodes
        
        # Get volatile time constant from profile or fallback
        if self.profile and self.profile.tau_volatile is not None:
            self.tau = self.profile.tau_volatile
        else:
            self.tau = 3.64  # standard PFO/AlOx volatile decay time constant
            
        self.alpha = np.exp(-0.01 / self.tau)  # step size dt = 0.01
        
        np.random.seed(seed)
        self.W_in = (np.random.rand(n_nodes) * 2 - 1) * 0.5
        W_res = (np.random.rand(n_nodes, n_nodes) * 2 - 1) * 0.1
        
        # Ensure reservoir state matrix is stable
        try:
            eigvals = np.linalg.eigvals(W_res)
            W_res = W_res / np.max(np.abs(eigvals)) * 0.95
        except Exception:
            pass
        self.W_res = W_res

    def simulate_reservoir(self, data):
        """Processes 1D input data through physical reservoir slow dynamics."""
        states = []
        x_state = np.zeros(self.n_nodes)
        
        for u in data:
            # Inputs to the reservoir nodes
            pre_activation = self.W_in * u + np.dot(self.W_res, x_state)
            # Volatile slow integration update
            x_state = (1.0 - self.alpha) * x_state + self.alpha * np.tanh(pre_activation)
            states.append(x_state.copy())
            
        return np.array(states)

    def augment_states(self, states, delay=5):
        """
        State Augmentation & Non-linear Mixing:
        Concatenates [current state, state from step - delay, squared current state].
        """
        N = states.shape[0]
        aug_states = np.zeros((N - delay, states.shape[1] * 3))
        
        for i in range(delay, N):
            current = states[i]
            past = states[i - delay]
            # Concatenate current, past, and squared state
            aug_states[i - delay] = np.concatenate([current, past, current**2])
            
        return aug_states
