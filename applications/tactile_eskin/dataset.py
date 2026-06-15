import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def generate_tactile_data(num_samples=600, time_steps=100, n_channels=16, noise_level=0.1):
    """
    Generates a simulated 16-channel flexible electronic skin dataset for 10 Braille characters (0-9).
    Each character is represented by a specific spatio-temporal excitation pattern 
    simulating finger sliding over tactile dots, with random speed variations and friction noise.
    """
    np.random.seed(42)
    X = []
    y = []
    
    # 10 Braille templates (each activates a subset of channels at different times)
    templates = {}
    for digit in range(10):
        # Deterministic but pseudo-random selection based on digit for reproducibility
        state = np.random.RandomState(digit + 100)
        active_channels = state.choice(n_channels, size=state.randint(4, 7), replace=False)
        centers = state.randint(15, 85, size=len(active_channels))
        widths = state.randint(8, 20, size=len(active_channels))
        templates[digit] = list(zip(active_channels, centers, widths))
        
    for i in range(num_samples):
        # We also use random seed derived from sample index to guarantee reproducibility
        state = np.random.RandomState(i)
        digit = state.randint(0, 10)
        sample = np.zeros((time_steps, n_channels))
        
        # Speed variation (temporal scaling)
        speed_factor = state.uniform(0.8, 1.25)
        
        for ch, center, width in templates[digit]:
            scaled_center = int(center * speed_factor)
            scaled_width = int(width * speed_factor)
            
            t = np.arange(time_steps)
            pulse = np.exp(-0.5 * ((t - scaled_center) / (scaled_width + 1e-5))**2)
            amp = state.uniform(0.7, 1.3)
            sample[:, ch] += amp * pulse
            
        # Add friction noise
        friction_noise = state.normal(0, noise_level, (time_steps, n_channels))
        sample += friction_noise
        sample = np.clip(sample, 0.0, 2.0)
        
        X.append(sample)
        y.append(digit)
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    return X, y

def get_tactile_dataloader(batch_size=32, noise_level=0.1):
    X, y = generate_tactile_data(num_samples=600, noise_level=noise_level)
    
    # Split 500 train, 100 test
    X_train, X_test = X[:500], X[500:]
    y_train, y_test = y[:500], y[500:]
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
