import os
import sys
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class GraspEnvironment:
    """
    Simulation of the physical object gripping environment.
    - F_ext: external pulling shear force (N)
    - F_grip: motor normal gripping force (N)
    - Friction coefficient: mu = 0.3
    - Slip velocity: v = max(0, F_ext - mu * F_grip) (cm/s)
    - Slip distance: d = integral(v * dt) (cm)
    """
    def __init__(self, mu=0.3, dt=0.001):
        self.mu = mu
        self.dt = dt
        self.slip_velocity = 0.0
        self.slip_distance = 0.0
        
    def reset(self):
        self.slip_velocity = 0.0
        self.slip_distance = 0.0
        
    def step(self, F_grip, F_ext):
        # Friction force limit
        F_friction = self.mu * F_grip
        
        if F_ext > F_friction:
            # Slipping occurs (accel scaling factor = 5.0)
            self.slip_velocity = (F_ext - F_friction) * 5.0
        else:
            self.slip_velocity = 0.0
            
        self.slip_distance += self.slip_velocity * self.dt
        return self.slip_velocity, self.slip_distance

class MemristorGraspReflexController:
    """
    Neuromorphic Reflex Controller using FingerMemristor (28-state) to adapt grip force.
    - Tactile slip velocity acts as the pre-synaptic pulse.
    - Conductance G maps to gripping force: F_grip = F_base + G_norm * (F_max - F_base)
    - Adapt update: delta G = eta * slip_velocity
    """
    def __init__(self, profile, eta=2.5, dt=0.001):
        self.profile = profile
        self.eta = eta
        self.dt = dt
        
        self.G_min = profile.conductance_min
        self.G_max = profile.conductance_max
        self.G_range = self.G_max - self.G_min
        
        # Base forces (N)
        self.F_base = 3.0
        self.F_max = 25.0
        
        # Initialize conductance to G_min (weakest grip)
        self.G = self.G_min
        self.G = self.quantize_G(self.G)
        
    def reset(self):
        self.G = self.G_min
        self.G = self.quantize_G(self.G)
        
    def quantize_G(self, G):
        if self.profile.discrete_states_count is not None:
            n_states = self.profile.discrete_states_count
            norm = (G - self.G_min) / self.G_range
            norm = np.clip(norm, 0.0, 1.0)
            quantized_norm = np.round(norm * (n_states - 1)) / (n_states - 1)
            return self.G_min + quantized_norm * self.G_range
        return G
        
    def get_grip_force(self):
        norm = (self.G - self.G_min) / self.G_range
        return self.F_base + norm * (self.F_max - self.F_base)
        
    def update_reflex(self, slip_velocity):
        if slip_velocity > 0:
            dG = self.eta * slip_velocity * self.G_range * self.dt
            
            # Apply polynomial non-linearity for LTP
            G_norm = (self.G - self.G_min) / self.G_range
            G_norm = np.clip(G_norm, 0.0, 1.0)
            coef = self.profile.ltp_poly_coefficients
            factor = coef[0] * (G_norm**3) + coef[1] * (G_norm**2) + coef[2] * G_norm + coef[3]
            
            dG_adjusted = dG * factor
            
            # C2C noise injection
            noise_std = self.profile.get_noise_std()
            if noise_std > 0:
                noise = np.random.normal(0, noise_std)
                dG_adjusted += noise
                
            self.G = np.clip(self.G + dG_adjusted, self.G_min, self.G_max)
            self.G = self.quantize_G(self.G)

class StaticGraspController:
    """
    Fixed-grip controller for baseline comparison.
    Uses a standard medium grip force.
    """
    def __init__(self, F_grip=10.0):
        self.F_grip = F_grip
        
    def reset(self):
        pass
        
    def get_grip_force(self):
        return self.F_grip
        
        # Does not adapt
    def update_reflex(self, slip_velocity):
        pass
