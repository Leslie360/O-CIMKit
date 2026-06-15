import os
import sys
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class DCMotor:
    """
    Simulation of a 2nd-order DC Motor physical system.
    State equations:
      di/dt = (V - R*i - Ke*omega) / L
      domega/dt = (Kt*i - B*omega - T_L) / J
    """
    def __init__(self, R=1.0, L=0.5, Kt=0.1, Ke=0.1, J=0.01, B=0.1, dt=0.001):
        self.R = R
        self.L = L
        self.Kt = Kt
        self.Ke = Ke
        self.J = J
        self.B = B
        self.dt = dt
        
        # State variables
        self.i = 0.0      # Armature current (A)
        self.omega = 0.0  # Rotor speed (rad/s)
        
    def reset(self):
        self.i = 0.0
        self.omega = 0.0
        
    def step(self, V, T_L=0.0):
        # Input voltage limit (saturation)
        V = np.clip(V, -24.0, 24.0)
        
        # Euler integration
        di = (V - self.R * self.i - self.Ke * self.omega) / self.L
        domega = (self.Kt * self.i - self.B * self.omega - T_L) / self.J
        
        self.i += di * self.dt
        self.omega += domega * self.dt
        
        return self.omega

class MemristorPIDController:
    """
    Hardware-Aware Memristive Adaptive PID Controller.
    PID gains (Kp, Ki, Kd) are mapped onto physical memristor conductances.
    Self-tuning update is performed online using error gradients, mapping changes 
    to LTP/LTD write pulses on the memristor device with non-linearities, 
    finite states (64-state quantization), and Cycle-to-Cycle (C2C) write noise.
    """
    def __init__(self, profile, Kp_bounds=(0.1, 8.0), Ki_bounds=(0.0, 15.0), Kd_bounds=(0.0, 2.0),
                 eta_p=0.2, eta_i=0.05, eta_d=0.005, dt=0.001):
        self.profile = profile
        self.Kp_bounds = Kp_bounds
        self.Ki_bounds = Ki_bounds
        self.Kd_bounds = Kd_bounds
        self.eta_p = eta_p
        self.eta_i = eta_i
        self.eta_d = eta_d
        self.dt = dt
        
        # Physical device parameters
        self.G_min = profile.conductance_min
        self.G_max = profile.conductance_max
        self.G_range = self.G_max - self.G_min
        
        # Initial mathematical gains (set to low/un-tuned values)
        Kp_init = 0.5
        Ki_init = 0.2
        Kd_init = 0.01
        
        # Map to physical conductances
        self.Gp = self.map_K_to_G(Kp_init, self.Kp_bounds)
        self.Gi = self.map_K_to_G(Ki_init, self.Ki_bounds)
        self.Gd = self.map_K_to_G(Kd_init, self.Kd_bounds)
        
        # Apply quantization on initialization
        self.Gp = self.quantize_G(self.Gp)
        self.Gi = self.quantize_G(self.Gi)
        self.Gd = self.quantize_G(self.Gd)
        
        # Controller internal state
        self.prev_error = 0.0
        self.integral = 0.0
        
    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        
        # Reset to initial gains
        Kp_init = 0.5
        Ki_init = 0.2
        Kd_init = 0.01
        
        self.Gp = self.quantize_G(self.map_K_to_G(Kp_init, self.Kp_bounds))
        self.Gi = self.quantize_G(self.map_K_to_G(Ki_init, self.Ki_bounds))
        self.Gd = self.quantize_G(self.map_K_to_G(Kd_init, self.Kd_bounds))
        
    def map_K_to_G(self, K, bounds):
        K_min, K_max = bounds
        norm = (K - K_min) / (K_max - K_min)
        norm = np.clip(norm, 0.0, 1.0)
        return self.G_min + norm * self.G_range
        
    def map_G_to_K(self, G, bounds):
        K_min, K_max = bounds
        norm = (G - self.G_min) / self.G_range
        norm = np.clip(norm, 0.0, 1.0)
        return K_min + norm * (K_max - K_min)
        
    def quantize_G(self, G):
        if self.profile.discrete_states_count is not None:
            n_states = self.profile.discrete_states_count
            norm = (G - self.G_min) / self.G_range
            norm = np.clip(norm, 0.0, 1.0)
            quantized_norm = np.round(norm * (n_states - 1)) / (n_states - 1)
            return self.G_min + quantized_norm * self.G_range
        return G
        
    def get_PID_gains(self):
        Kp = self.map_G_to_K(self.Gp, self.Kp_bounds)
        Ki = self.map_G_to_K(self.Gi, self.Ki_bounds)
        Kd = self.map_G_to_K(self.Gd, self.Kd_bounds)
        return Kp, Ki, Kd
        
    def update_memristor_conductance(self, G, dG):
        if dG == 0:
            return G
            
        G_norm = (G - self.G_min) / self.G_range
        G_norm = np.clip(G_norm, 0.0, 1.0)
        
        # Apply polynomial fitting scaling for non-linearity
        if dG > 0:
            coef = self.profile.ltp_poly_coefficients
            factor = coef[0] * (G_norm**3) + coef[1] * (G_norm**2) + coef[2] * G_norm + coef[3]
        else:
            coef = self.profile.ltd_poly_coefficients
            factor = coef[0] * (G_norm**3) + coef[1] * (G_norm**2) + coef[2] * G_norm + coef[3]
            
        # Adjusted physical delta
        dG_adjusted = dG * factor
        
        # Cycle-to-Cycle (C2C) write noise injection
        noise_std = self.profile.get_noise_std()
        if noise_std > 0:
            noise = np.random.normal(0, noise_std)
            dG_adjusted += noise
            
        G_new = G + dG_adjusted
        G_new = np.clip(G_new, self.G_min, self.G_max)
        
        # Quantize conductance to discrete states (e.g. 64 states)
        G_new = self.quantize_G(G_new)
        return G_new
        
    def compute(self, error, update=True):
        self.integral += error * self.dt
        # Clamp integral error to prevent integrator windup
        self.integral = np.clip(self.integral, -50.0, 50.0)
        
        derivative = (error - self.prev_error) / self.dt
        
        Kp, Ki, Kd = self.get_PID_gains()
        u = Kp * error + Ki * self.integral + Kd * derivative
        
        if update:
            # Gradient descent heuristic for PID adaptive tuning:
            # dK_j = eta * e * x_j (using sign(dy/du) > 0)
            dKp = self.eta_p * error * error
            dKi = self.eta_i * error * self.integral
            dKd = self.eta_d * error * derivative
            
            # Map math parameter delta to physical conductance delta
            dGp = dKp * self.G_range / (self.Kp_bounds[1] - self.Kp_bounds[0])
            dGi = dKi * self.G_range / (self.Ki_bounds[1] - self.Ki_bounds[0])
            dGd = dKd * self.G_range / (self.Kd_bounds[1] - self.Kd_bounds[0])
            
            # Apply memristive physical updates
            self.Gp = self.update_memristor_conductance(self.Gp, dGp)
            self.Gi = self.update_memristor_conductance(self.Gi, dGi)
            self.Gd = self.update_memristor_conductance(self.Gd, dGd)
            
        self.prev_error = error
        return u

class BaselinePIDController:
    """
    Standard Static PID Controller (un-tuned baseline).
    """
    def __init__(self, Kp=0.5, Ki=0.2, Kd=0.01, dt=0.001):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        
        self.prev_error = 0.0
        self.integral = 0.0
        
    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        
    def compute(self, error):
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -50.0, 50.0)
        
        derivative = (error - self.prev_error) / self.dt
        
        u = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return u
