import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.layers import QATMLPLayer

class BearingMultiScaleReservoir(nn.Module):
    """
    Multi-Scale Reservoir Computing Layer for CWRU bearing vibration signals.
    """
    def __init__(self, taus=[3, 10, 30], n_reservoir=500, n_inputs=1, seed=42, device='cpu'):
        super().__init__()
        self.n_reservoir = n_reservoir
        self.device = device
        
        self.reservoirs = []
        for i, tau in enumerate(taus):
            torch.manual_seed(seed + i)
            W_in = torch.randn(n_reservoir, n_inputs, device=device) * 0.5
            W = torch.randn(n_reservoir, n_reservoir, device=device) * 0.1
            mask = torch.rand(n_reservoir, n_reservoir, device=device) < 0.1
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
                'alpha': torch.tensor(1.0 / tau, device=device)
            })

    def forward(self, X_batch):
        """
        Processes batch of vibration signals.
        Args:
            X_batch (torch.Tensor): Vibration input shape (batch_size, seq_len, n_inputs).
        Returns:
            torch.Tensor: Concatenated features (batch_size, len(taus) * (4 + 50)).
        """
        batch_size, seq_len, _ = X_batch.shape
        all_features = []
        
        for res in self.reservoirs:
            states = torch.zeros(batch_size, self.n_reservoir, device=self.device)
            for t in range(seq_len):
                x = X_batch[:, t, :]
                pre = res['W_in'] @ x.T + res['W'] @ states.T
                alpha = res['alpha']
                states = (1.0 - alpha) * states + alpha * torch.tanh(pre.T)
                
            # Feature extraction matching PhD's: mean, std, max, min, and first 50 states
            feat = torch.cat([
                states.mean(dim=1, keepdim=True),
                states.std(dim=1, keepdim=True),
                states.max(dim=1, keepdim=True)[0],
                states.min(dim=1, keepdim=True)[0],
                states[:, :50],
            ], dim=1)
            all_features.append(feat)
            
        return torch.cat(all_features, dim=1)

class BearingQATClassifier(nn.Module):
    """
    Bearing Readout MLP Classifier using QATMLPLayer mapping weights to
    discrete physical states (e.g. 64-states) specified by DeviceProfile.
    """
    def __init__(self, in_features, hidden_dim, out_classes, device_profile=None, dropout_p=0.3):
        super().__init__()
        self.fc1 = QATMLPLayer(in_features, hidden_dim, device_profile=device_profile)
        self.fc2 = QATMLPLayer(hidden_dim, out_classes, device_profile=device_profile)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, x):
        h = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(h)
