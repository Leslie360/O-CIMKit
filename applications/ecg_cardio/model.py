import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.layers import QATMLPLayer

class OptoelectronicReservoir(nn.Module):
    """
    Optoelectronic Dual-Input physical reservoir for ECG classification,
    modulating electrical signal and optical signal pathways.
    """
    def __init__(self, device_profile, n_inputs=1, n_reservoir=2000, seed=42, device='cpu'):
        super().__init__()
        self.profile = device_profile
        self.n_reservoir = n_reservoir
        self.device = device
        
        # Get volatile leaking rate
        if self.profile and self.profile.is_volatile:
            # Match PhD update equation: alpha = exp(-1/tau)
            # In our device_profile, leaking_rate_volatile is 1 - exp(-1/tau)
            # Let's compute alpha_phd = exp(-1/tau)
            self.alpha = float(np.exp(-1.0 / self.profile.tau_volatile))
        else:
            self.alpha = float(np.exp(-1.0 / 3.64))  # default for tau=3.64s
            
        torch.manual_seed(seed)
        
        # Input coupling matrices
        self.W_elec = torch.randn(n_reservoir, n_inputs, device=device) * 0.5
        self.W_opt = torch.randn(n_reservoir, n_inputs, device=device) * 0.5
        
        # Internal connection matrix
        W = torch.randn(n_reservoir, n_reservoir, device=device) * 0.1
        mask = torch.rand(n_reservoir, n_reservoir, device=device) < 0.05
        W = W * mask
        
        # Scale spectral radius
        try:
            eig = torch.linalg.eigvals(W[:500, :500]).abs().max()
            if eig > 0:
                W = W / eig * 0.95
        except Exception:
            pass
        self.W = W

    def forward(self, X_batch):
        """
        Args:
            X_batch (torch.Tensor): ECG input batch of shape (batch_size, seq_len).
        Returns:
            torch.Tensor: State history of shape (batch_size, seq_len, n_reservoir).
        """
        batch_size, seq_len = X_batch.shape
        states = []
        state = torch.zeros(batch_size, self.n_reservoir, device=self.device)
        
        # Compute smooth optical input (moving average pool of size 5)
        X_smooth = torch.nn.functional.avg_pool1d(
            X_batch.unsqueeze(1).float(), kernel_size=5, stride=1, padding=2
        ).squeeze(1)
        
        for t in range(seq_len):
            x_e = X_batch[:, t].unsqueeze(1)
            x_o = X_smooth[:, t].unsqueeze(1)
            
            # Linear input combination + reservoir feedback
            pre = (x_e @ self.W_elec.T) + (x_o @ self.W_opt.T) + (state @ self.W.T)
            
            # PhD's update equation: x(t) = (1-alpha)*x(t-1) + alpha*tanh(pre)
            state = (1.0 - self.alpha) * state + self.alpha * torch.tanh(pre)
            states.append(state.clone())
            
        return torch.stack(states, dim=1)

    def extract_features(self, states_tensor):
        """
        Extract high-dimensional features from states history.
        Concats: Mean, Max, Min, Std, and Final state.
        Output shape: (batch_size, 5 * n_reservoir).
        """
        mean_feat = states_tensor.mean(dim=1)
        max_feat = states_tensor.max(dim=1)[0]
        min_feat = states_tensor.min(dim=1)[0]
        std_feat = states_tensor.std(dim=1)
        final_feat = states_tensor[:, -1, :]
        return torch.cat([mean_feat, max_feat, min_feat, std_feat, final_feat], dim=1)

class ECGQATClassifier(nn.Module):
    """
    ECG Readout MLP Classifier using QATMLPLayer mapping weights to
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
