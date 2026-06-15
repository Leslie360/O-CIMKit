import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.layers import QATMLPLayer

class BiohybridOECTSpiker(nn.Module):
    """
    Biohybrid Neuromorphic Spiking Encoder using OECT mixed ionic-electronic physics 
    and a Population-Coded Leaky Integrate-and-Fire (LIF) spike generator.
    Supports a pure software float baseline mode where standard nn.Linear 
    is instantiated for the classification layer instead of QATMLPLayer.
    """
    def __init__(self, alox_profile, n_neurons=24, tau_oect=8.0, G0=0.1, beta=1.8, leak_lif=0.1, dt=1.0, is_float_baseline=False):
        super().__init__()
        self.tau_oect = tau_oect
        self.alpha_oect = dt / tau_oect
        self.G0 = G0
        self.beta = beta
        self.leak_lif = leak_lif
        self.n_neurons = n_neurons
        self.is_float_baseline = is_float_baseline
        
        # 16/24 neurons with uniform thresholds from 0.2 to 2.5 V
        self.register_buffer("V_th", torch.linspace(0.2, 2.5, n_neurons))
        
        if is_float_baseline:
            # Ideal float FC layer
            self.fc = nn.Linear(in_features=n_neurons, out_features=10)
        else:
            # QAT linear classifier
            self.fc = QATMLPLayer(
                in_features=n_neurons,
                out_features=10,  # 10 classes (labels 0-9)
                device_profile=alox_profile,
                mode="minmax"
            )
        
    def forward(self, x):
        # x shape: (batch_size, time_steps, 1)
        batch_size, time_steps, _ = x.shape
        device = x.device
        
        # Initial states
        G = torch.zeros(batch_size, device=device) + self.G0
        V_mem = torch.zeros(batch_size, self.n_neurons, device=device)
        spike_counts = torch.zeros(batch_size, self.n_neurons, device=device)
        
        # Simulate temporal dynamics
        for t in range(time_steps):
            V_in = x[:, t, 0]  # (batch_size,)
            
            # OECT Ion-Electron mixed conduction relaxation
            G = G * (1.0 - self.alpha_oect) + self.alpha_oect * (self.G0 + self.beta * V_in)
            
            # Channel current injection (V_ds = 0.5 V)
            I_channel = G * 0.5  # (batch_size,)
            
            # LIF integration with population coding broadcast
            V_mem = V_mem * (1.0 - self.leak_lif) + I_channel.unsqueeze(1)
            
            # Fire spikes based on individual neuron thresholds
            is_firing = (V_mem >= self.V_th).float()
            spike_counts += is_firing
            
            # Reset
            V_mem = V_mem * (1.0 - is_firing)
            
        # Feature vector: normalized firing rate per neuron (shape: batch_size, n_neurons)
        features = spike_counts / float(time_steps)
        
        # Classification
        out = self.fc(features)
        return out
