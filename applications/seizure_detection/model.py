import os
import sys
import torch
import torch.nn as nn

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.layers import OrganicSynapseConv, QATMLPLayer

class SeizureCIMClassifier(nn.Module):
    """
    Hardware-Aware Convolutional Network for Seizure Detection.
    Supports a pure software float baseline mode where standard nn.Conv2d 
    and nn.Linear are instantiated to avoid profile dependencies.
    """
    def __init__(self, memristor_profile, alox_profile, n_channels=18, out_channels=16, kernel_size=9, is_float_baseline=False):
        super().__init__()
        self.is_float_baseline = is_float_baseline
        
        if is_float_baseline:
            # Standard ideal float Conv/FC layers
            self.conv = nn.Conv2d(
                in_channels=n_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, 1)
            )
            self.fc = nn.Linear(in_features=out_channels, out_features=2)
        else:
            # Hardware-constrained layers
            self.conv = OrganicSynapseConv(
                in_channels=n_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, 1),
                device_profile=memristor_profile
            )
            self.fc = QATMLPLayer(
                in_features=out_channels,
                out_features=2,
                device_profile=alox_profile,
                mode="minmax"
            )
            
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, time_steps, n_channels)
        # Permute to (batch_size, n_channels, time_steps)
        x = x.permute(0, 2, 1)
        # Add dummy height dimension: (batch_size, n_channels, time_steps, 1)
        x = x.unsqueeze(3)
        
        # Conv feature extraction
        x = self.conv(x)
        
        # Remove dummy dimension: (batch_size, out_channels, out_time_steps)
        x = x.squeeze(3)
        
        # Pool & activate
        x = self.pool(x).squeeze(2)
        x = self.relu(x)
        
        # Classification readout
        out = self.fc(x)
        return out
