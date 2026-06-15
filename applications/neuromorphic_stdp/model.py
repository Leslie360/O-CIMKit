import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class OrganicUnsupervisedSNN(nn.Module):
    """
    Hardware-Aware Spiking Neural Network (SNN) with Unsupervised STDP Learning,
    Homeostasis (Adaptive Threshold), Synaptic Scaling, and Global Soft Inhibition.
    """
    def __init__(self, n_inputs=64, n_neurons=30, device_profile=None, 
                 tau_m=15.0, V_th=0.4, V_reset=0.0, 
                 eta=0.2, tau_plus=20.0, tau_minus=20.0, dt=1.0):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.profile = device_profile
        
        # Hyperparameters
        self.tau_m = tau_m
        self.V_th = V_th
        self.V_reset = V_reset
        self.alpha_leak = dt / tau_m
        self.eta = eta
        self.dt = dt
        self.decay_plus = np.exp(-dt / tau_plus)
        self.decay_minus = np.exp(-dt / tau_minus)
        
        # Excitatory weights initialized to positive random values in [0.05, 0.2] with diversity
        torch.manual_seed(42)
        self.weight = nn.Parameter(torch.rand(n_inputs, n_neurons) * 0.15 + 0.05)
        
        # Device non-idealities
        if self.profile:
            self.noise_std = self.profile.get_noise_std()
        else:
            self.noise_std = 0.0

    def forward(self, spikes):
        """
        Args:
            spikes (torch.Tensor): Poisson input spike trains of shape (batch_size, time_window, n_inputs)
        Returns:
            torch.Tensor: Firing spike counts of shape (batch_size, n_neurons)
        """
        batch_size, time_window, _ = spikes.shape
        device = spikes.device
        
        # State variables
        V = torch.zeros(batch_size, self.n_neurons, device=device)
        x_trace = torch.zeros(batch_size, self.n_inputs, device=device)
        y_trace = torch.zeros(batch_size, self.n_neurons, device=device)
        
        # Homeostasis (Adaptive Threshold)
        theta = torch.zeros(batch_size, self.n_neurons, device=device)
        
        # Cumulative output spikes count
        spike_counts = torch.zeros(batch_size, self.n_neurons, device=device)
        
        # Weight delta accumulation per batch
        delta_w = torch.zeros(self.n_inputs, self.n_neurons, device=device)
        
        for t in range(time_window):
            x_t = spikes[:, t, :] # (batch_size, n_inputs)
            
            # Trace dynamics
            x_trace = x_trace * self.decay_plus + x_t
            y_trace = y_trace * self.decay_minus
            
            # Integration of input currents
            I_t = x_t @ self.weight
            V = V * (1.0 - self.alpha_leak) + I_t
            
            # Global Soft Inhibition: subtract 25% of the mean potential of all neurons 
            # to amplify the contrast between highly matched neurons and losers
            V = V - 0.25 * V.mean(dim=1, keepdim=True)
            V = torch.clamp(V, min=0.0) # Membrane potential cannot go negative
            
            # Firing threshold evaluation with adaptive offset
            effective_V_th = self.V_th + theta
            is_firing = (V >= effective_V_th).float()
            
            # WTA competition based on excess voltage
            winner_mask = torch.zeros_like(V)
            excess_v = V - effective_V_th
            max_vals, max_inds = torch.max(excess_v, dim=1)
            above_th = (max_vals >= 0.0).float()
            winner_mask.scatter_(1, max_inds.unsqueeze(1), 1.0)
            winner_mask = winner_mask * above_th.unsqueeze(1)
            
            # Update outputs and homeostasis thresholds
            spike_counts += winner_mask
            
            if self.training:
                theta = theta + 0.15 * winner_mask # Elevate threshold of winner
                
            y_trace = y_trace + winner_mask
            
            # Reset dynamics of firing winners & losers
            V = V * (1.0 - winner_mask) + self.V_reset * winner_mask
            V = V * (1.0 - is_firing * (1.0 - winner_mask)) + self.V_reset * is_firing * (1.0 - winner_mask)
            
            # Decay adaptive threshold
            theta = theta * 0.95
            
            # Unsupervised Trace STDP Learning
            if self.training:
                # LTP
                ltp_update = self.eta * (x_trace.T @ winner_mask)
                # LTD (scaled to 50% of LTP for memory preservation)
                ltd_update = -0.5 * self.eta * (x_t.T @ y_trace)
                delta_w += (ltp_update + ltd_update)

        # Apply unsupervised hardware weight updates & Synaptic Scaling
        if self.training:
            # Normalize STDP updates by batch size to prevent weight explosion
            delta_w = delta_w / batch_size
            
            # Map weights [-1, 1] to [0, 1] for polynomial curve lookup
            w_norm = torch.clamp((self.weight.data + 1.0) / 2.0, 0.0, 1.0)
            
            if self.profile and self.profile.ltp_poly_coefficients:
                ltp_coef = self.profile.ltp_poly_coefficients
                ltp_factor = ltp_coef[0] * (w_norm**3) + ltp_coef[1] * (w_norm**2) + ltp_coef[2] * w_norm + ltp_coef[3]
            else:
                ltp_factor = torch.ones_like(self.weight.data)
                
            if self.profile and self.profile.ltd_poly_coefficients:
                ltd_coef = self.profile.ltd_poly_coefficients
                ltd_factor = ltd_coef[0] * (w_norm**3) + ltd_coef[1] * (w_norm**2) + ltd_coef[2] * w_norm + ltd_coef[3]
            else:
                ltd_factor = torch.ones_like(self.weight.data)
                
            # Apply polynomial device non-linear updates
            adjusted_delta = torch.where(
                delta_w > 0,
                delta_w * ltp_factor,
                delta_w * ltd_factor
            )
            
            self.weight.data += adjusted_delta
            
            # Inject C2C write noise
            if self.noise_std > 0:
                noise = torch.randn_like(self.weight.data) * self.noise_std
                self.weight.data += noise
                
            # Excitatory weight clamp in [0.0, 1.0]
            self.weight.data.clamp_(0.0, 1.0)
            
            # Synaptic Scaling: Keep weight sum per neuron constant at 3.5
            w_sum = self.weight.data.sum(dim=0, keepdim=True)
            self.weight.data = self.weight.data / (w_sum + 1e-8) * 3.5
            
        return spike_counts
