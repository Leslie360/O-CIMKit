import os
import sys
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def get_aigc_dataloaders(batch_size=32):
    """
    Loads sklearn 8x8 digits dataset, reshapes to (1, 8, 8),
    and returns PyTorch DataLoaders.
    """
    digits = load_digits()
    X, y = digits.images, digits.target
    
    # Scale inputs to [0, 1]
    X = X / 16.0
    
    # Reshape to (N, 1, 8, 8)
    X = np.expand_dims(X, axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
