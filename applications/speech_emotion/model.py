import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.layers import QATMLPLayer

class MultiScaleAttentionReservoir(nn.Module):
    """
    Multi-Scale Reservoir with multi-head self-attention mechanisms
    for Speech MFCC features decoding.
    """
    def __init__(self, taus=[2.0, 5.0, 10.0, 20.0, 40.0], n_res=400, n_heads=4, n_inputs=39, seed=42, device='cpu'):
        super().__init__()
        self.n_res = n_res
        self.n_heads = n_heads
        self.d_k = n_res // n_heads
        self.taus = taus
        self.device = device
        
        self.reservoirs = []
        for i, tau in enumerate(taus):
            torch.manual_seed(seed + i)
            W_in = torch.randn(n_res, n_inputs, device=device) * 0.3
            W = torch.randn(n_res, n_res, device=device) * 0.1
            mask = torch.rand(n_res, n_res, device=device) < 0.05
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
        """
        Processes batch and performs self-attention fusion on state histories.
        Returns:
            torch.Tensor: Attentive features from all scales (batch_size, len(taus) * n_res * 4).
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
                
            # Multihead self-attention
            head_outputs = []
            for i in range(self.n_heads):
                h_states = states[:, :, i*self.d_k : (i+1)*self.d_k]
                scores = torch.bmm(h_states, h_states.transpose(1, 2)) / np.sqrt(self.d_k)
                attn = torch.softmax(scores, dim=-1)
                context = torch.bmm(attn, h_states)
                head_outputs.append(context)
                
            attn_states = torch.cat(head_outputs, dim=-1)
            fused = states + attn_states
            
            # Extract statistics over time axis (dim=1)
            feat_mean = fused.mean(dim=1)
            feat_std = fused.std(dim=1)
            feat_max = fused.max(dim=1)[0]
            feat_min = fused.min(dim=1)[0]
            
            scale_feat = torch.cat([feat_mean, feat_std, feat_max, feat_min], dim=1)
            all_scale_features.append(scale_feat)
            
        return torch.cat(all_scale_features, dim=1)

class SpeechQATClassifier(nn.Module):
    """
    Speech Emotion Readout MLP Classifier using QATMLPLayer mapping weights to
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
