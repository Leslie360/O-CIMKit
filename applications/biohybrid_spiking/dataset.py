import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def generate_biohybrid_data(num_samples=600, time_steps=100, noise_level=0.05):
    """
    Generates simulated bio-chemical concentration input patterns (binary/analog steps).
    - Represents 10 different transmitter stimulus concentrations (labels 0-9).
    - Returns input pattern array of shape (num_samples, time_steps, 1) and labels.
    """
    np.random.seed(42)
    X = []
    y = []
    
    for i in range(num_samples):
        state = np.random.RandomState(i)
        label = state.randint(0, 10)
        
        # Base amplitude maps to label
        amplitude = 0.2 + label * 0.18
        
        # Input step signal with temporal offset
        start_step = state.randint(10, 25)
        end_step = state.randint(75, 90)
        
        sample = np.zeros((time_steps, 1))
        sample[start_step:end_step, 0] = amplitude
        
        # Add local chemical fluctuation noise
        sample += state.normal(0, noise_level, (time_steps, 1))
        sample = np.clip(sample, 0.0, 2.5)
        
        X.append(sample)
        y.append(label)
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    return X, y

def get_biohybrid_dataloader(batch_size=32, noise_level=0.05):
    X, y = generate_biohybrid_data(num_samples=600, noise_level=noise_level)
    
    # Split 500 train, 100 test
    X_train, X_test = X[:500], X[500:]
    y_train, y_test = y[:500], y[500:]
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
