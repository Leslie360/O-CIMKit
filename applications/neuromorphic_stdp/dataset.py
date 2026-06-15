import os
import sys
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def poisson_spike_encoder(x, time_window=100, max_freq=100.0, dt=0.001):
    """
    Encodes analog values in [0, 1] into binary Poisson spike trains.
    Applies contrast filtering (thresholding at 0.25) to remove background noise.
    """
    batch_size, num_features = x.shape
    # Filter low-intensity noise
    x_filtered = torch.where(x > 0.25, x, torch.zeros_like(x))
    
    # Calculate firing probability at each step: P = freq * dt
    prob = x_filtered.unsqueeze(1).repeat(1, time_window, 1) * (max_freq * dt)
    
    # Sample binary spike events
    rand_vals = torch.rand(batch_size, time_window, num_features, device=x.device)
    spikes = (rand_vals < prob).float()
    return spikes

def get_stdp_dataset(batch_size=64, time_window=100, seed=42):
    """
    Loads 8x8 digit dataset, normalizes to [0, 1], splits train/test, 
    and returns PyTorch DataLoader.
    """
    digits = load_digits()
    X = digits.data / 16.0 # Normalize to [0, 1]
    y = digits.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = torch.utils.data.TensorDataset(X_test_t, y_test_t)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
