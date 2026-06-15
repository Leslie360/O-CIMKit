import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.layers import OrganicSynapseConv, QATMLPLayer

class GPUMultiScaleReservoir(nn.Module):
    """
    Multi-Scale Reservoir Computing Layer for multi-channel Sleep EEG signals,
    using custom time constants. (Retained for backward compatibility)
    """
    def __init__(self, taus=[3.0, 10.0, 30.0], n_res=300, n_inputs=1, seed=42, device='cpu'):
        super().__init__()
        self.n_res = n_res
        self.taus = taus
        self.device = device
        
        self.reservoirs = []
        for i, tau in enumerate(taus):
            torch.manual_seed(seed + i)
            W_in = torch.randn(n_res, n_inputs, device=device) * 0.5
            W = torch.randn(n_res, n_res, device=device) * 0.1
            mask = torch.rand(n_res, n_res, device=device) < 0.1
            W = W * mask
            
            try:
                eig = torch.linalg.eigvals(W[:200, :200]).abs().max()
                if eig > 0:
                    W = W / eig * 0.9
            except Exception:
                pass
                
            self.reservoirs.append({
                'W_in': W_in,
                'W': W,
                'alpha': torch.tensor(np.exp(-1.0 / tau), device=device)
            })

    def forward(self, X_batch):
        batch_size, seq_len, _ = X_batch.shape
        states_list = []
        
        for res in self.reservoirs:
            h = torch.zeros(batch_size, self.n_res, device=self.device)
            states = torch.zeros(batch_size, seq_len, self.n_res, device=self.device)
            for t in range(seq_len):
                pre = X_batch[:, t, :] @ res['W_in'].T + h @ res['W'].T
                alpha = res['alpha']
                h = (1.0 - alpha) * h + alpha * torch.tanh(pre)
                states[:, t, :] = h
            states_list.append(states)
            
        return torch.cat(states_list, dim=2)

    def extract_features(self, states):
        mean_feat = states.mean(dim=1)
        std_feat = states.std(dim=1)
        max_feat = states.max(dim=1)[0]
        min_feat = states.min(dim=1)[0]
        final_feat = states[:, -1, :]
        return torch.cat([mean_feat, std_feat, max_feat, min_feat, final_feat], dim=1)

class OrganicEEGNet(nn.Module):
    """
    End-to-End hardware-aware CNN for Sleep-EDF EEG fatigue detection.
    Replaces standard convolutions with OrganicSynapseConv and classifiers
    with QATMLPLayer to fully comply with memristor non-idealities and device noise.
    """
    def __init__(self, device_profile=None, num_classes=3, dropout_p=0.3):
        super().__init__()
        self.profile = device_profile
        
        # Temporal convolution (extract temporal frequency characteristics)
        self.conv1 = OrganicSynapseConv(
            in_channels=2, out_channels=16, kernel_size=(32, 1),
            device_profile=device_profile, stride=1, padding=(16, 0), bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        
        # Spatial Depthwise convolution to mix spatial channels
        self.conv2 = OrganicSynapseConv(
            in_channels=16, out_channels=32, kernel_size=(1, 1),
            device_profile=device_profile, stride=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.pool1 = nn.AvgPool2d(kernel_size=(4, 1))
        
        # Depthwise Separable Conv
        self.conv3 = OrganicSynapseConv(
            in_channels=32, out_channels=32, kernel_size=(16, 1),
            device_profile=device_profile, stride=1, padding=(8, 0), groups=32, bias=False
        )
        self.bn3 = nn.BatchNorm2d(32)
        
        # Pointwise Conv
        self.conv4 = OrganicSynapseConv(
            in_channels=32, out_channels=64, kernel_size=(1, 1),
            device_profile=device_profile, stride=1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=(8, 1))
        
        # Classifier Readout QAT Layers
        # input dim = 64 * 23 = 1472
        self.fc1 = QATMLPLayer(1472, 128, device_profile=device_profile)
        self.fc2 = QATMLPLayer(128, num_classes, device_profile=device_profile)
        
    def forward(self, x):
        # input x shape: (batch_size, 2, 750)
        x = x.unsqueeze(-1) # shape: (batch_size, 2, 750, 1)
        
        h1 = self.bn1(self.conv1(x))
        h2 = self.dropout(self.relu(self.bn2(self.conv2(h1))))
        h2 = self.pool1(h2)
        
        h3 = self.relu(self.bn3(self.conv3(h2)))
        h4 = self.dropout(self.relu(self.bn4(self.conv4(h3))))
        h4 = self.pool2(h4)
        
        flat = h4.view(h4.size(0), -1)
        
        out1 = self.dropout(self.relu(self.fc1(flat)))
        return self.fc2(out1)
