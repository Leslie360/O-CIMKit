import os
import sys
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def load_mnist_data(num_samples=1000, data_dir=None):
    """
    Loads sklearn 8x8 digits dataset to serve as the sMNIST benchmark.
    Flattens images to 64 vectors representing sequential optical signals.
    """
    print("📖 Loading sklearn 8x8 digits dataset for sMNIST...")
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Scale inputs to [0, 1]
    X_norm = X / 16.0
    
    # Limit to num_samples
    X_norm = X_norm[:num_samples]
    y = y[:num_samples]
    
    return torch.FloatTensor(X_norm), np.array(y)
