import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.layers import QATMLPLayer

class OlfactoryEnoseReservoir(nn.Module):
    """
    Volatile physical reservoir optimized for time-series gas sensor array responses.
    """
    def __init__(self, taus=[1.5, 3.0, 6.0], n_res=128, n_inputs=8, seed=42, device='cpu'):
        super().__init__()
        self.n_res = n_res
        self.taus = taus
        self.device = device
        
        self.reservoirs = []
        for i, tau in enumerate(taus):
            torch.manual_seed(seed + i)
            W_in = torch.randn(n_res, n_inputs, device=device) * 0.4
            W = torch.randn(n_res, n_res, device=device) * 0.1
            mask = torch.rand(n_res, n_res, device=device) < 0.1
            W = W * mask
            
            try:
                eig = torch.linalg.eigvals(W[:64, :64]).abs().max()
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
        """
        Processes batch and returns concatenated statistical features.
        """
        batch_size, seq_len, _ = X_batch.shape
        all_scale_features = []
        
        for res in self.reservoirs:
            h = torch.zeros(batch_size, self.n_res, device=self.device)
            states = torch.zeros(batch_size, seq_len, self.n_res, device=self.device)
            
            for t in range(seq_len):
                pre = X_batch[:, t, :] @ res['W_in'].T + h @ res['W'].T
                alpha = res['alpha']
                h = (1.0 - alpha) * h + alpha * torch.tanh(pre)
                states[:, t, :] = h
                
            # Extract statistical features over time axis (dim=1)
            feat_mean = states.mean(dim=1)
            feat_std = states.std(dim=1)
            feat_max = states.max(dim=1)[0]
            feat_min = states.min(dim=1)[0]
            
            scale_feat = torch.cat([feat_mean, feat_std, feat_max, feat_min], dim=1)
            all_scale_features.append(scale_feat)
            
        return torch.cat(all_scale_features, dim=1)

class OlfactoryQATClassifier(nn.Module):
    """
    Readout QAT MLP for gas classification, mapping weights to discrete device states.
    """
    def __init__(self, in_features, hidden_dim, out_classes, device_profile=None, dropout_p=0.2):
        super().__init__()
        self.fc1 = QATMLPLayer(in_features, hidden_dim, device_profile=device_profile)
        self.fc2 = QATMLPLayer(hidden_dim, out_classes, device_profile=device_profile)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, x):
        h = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(h)
