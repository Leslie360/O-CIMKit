import numpy as np
import torch
import torch.nn as nn

class VolatileReservoir:
    """Volatile reservoir (fast state, processing dynamic friction/pressure)."""
    def __init__(self, tau=3.64, sigma=0.01):
        self.tau = tau
        self.sigma = sigma
        self.alpha = np.exp(-1.0 / tau)
        self.state = 0.0

    def reset(self):
        self.state = 0.0

    def step(self, x_t):
        u_t = self.alpha * self.state + (1.0 - self.alpha) * x_t
        u_t += self.sigma * np.random.randn()
        self.state = np.clip(u_t, -10.0, 10.0)
        return self.state

class NonVolatileMemory:
    """Non-volatile memory (slow state, storing material baseline)."""
    def __init__(self, n_states=64):
        self.n_states = n_states
        self.states = np.zeros(n_states)
        self.current_idx = 0

    def write(self, value):
        """Write state quantized to the nearest level."""
        quantized = np.round(value * (self.n_states - 1)) / (self.n_states - 1)
        self.states[self.current_idx] = quantized
        self.current_idx = (self.current_idx + 1) % self.n_states

    def read(self):
        return self.states.copy()

class EmbodiedAIRC:
    """
    Embodied AI Reservoir Computing using dual modes:
    - Volatile (tau = 3.64s): processing dynamic friction/pressure during sliding.
    - Non-volatile (64 states): storing material baseline memory.
    """
    def __init__(self, device_profile=None, tau=3.64, n_states=64):
        self.profile = device_profile
        
        # Load parameters from device profile if available
        if self.profile:
            tau = self.profile.tau_volatile if self.profile.tau_volatile is not None else tau
            n_states = self.profile.discrete_states_count if self.profile.discrete_states_count is not None else n_states
            
        self.volatile = VolatileReservoir(tau)
        self.nonvolatile = NonVolatileMemory(n_states)

    def process_tactile_signal(self, signal, material_baseline=None):
        self.volatile.reset()
        n_samples = len(signal)
        states = np.zeros((n_samples, 2))

        for t in range(n_samples):
            states[t, 0] = self.volatile.step(signal[t])
            if material_baseline is not None:
                # Apply non-volatile quantization if profile specifies
                if self.profile and self.profile.discrete_states_count is not None:
                    n_states = self.profile.discrete_states_count
                    baseline = np.round(material_baseline * (n_states - 1)) / (n_states - 1)
                else:
                    baseline = material_baseline
                states[t, 1] = baseline
            else:
                states[t, 1] = 0.0

        return states
