import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def generate_seizure_data(num_samples=500, time_steps=128, n_channels=18, noise_level=0.2):
    """
    Generates a simulated 18-channel EEG dataset for seizure detection (binary classification).
    - Label 0: Normal brain activity (low-amplitude high-frequency noise, 1/f background).
    - Label 1: Seizure activity (high-amplitude synchronized 3Hz spike-and-wave discharges).
    """
    np.random.seed(42)
    X = []
    y = []
    
    t = np.linspace(0, 4.0, time_steps)  # 4 seconds epoch
    
    for i in range(num_samples):
        state = np.random.RandomState(i)
        label = state.randint(0, 2)
        sample = np.zeros((time_steps, n_channels))
        
        # 1/f background noise
        for ch in range(n_channels):
            # Generate random noise and perform a basic low-pass filter to simulate 1/f eeg spectrum
            raw_noise = state.normal(0, 1.0, time_steps)
            filtered_noise = np.convolve(raw_noise, np.ones(5)/5.0, mode='same')
            sample[:, ch] = filtered_noise * 0.3
            
        if label == 1:
            # Seizure: Synchronized high-amplitude 3Hz spikes
            # Create a 3Hz spike wave template
            spike_wave = np.sin(2 * np.pi * 3.0 * t)
            # Make it look like a sharp spike by powering odd exponents
            spike_wave = np.sign(spike_wave) * (np.abs(spike_wave) ** 0.4)
            
            # Apply to a random subset of 12-16 synchronized channels
            seizure_channels = state.choice(n_channels, size=state.randint(12, 17), replace=False)
            
            # Random phase offset per channel to add physical propagation lag
            for ch in seizure_channels:
                phase_shift = state.uniform(-0.15, 0.15)
                shifted_spike = np.sin(2 * np.pi * 3.0 * (t + phase_shift))
                shifted_spike = np.sign(shifted_spike) * (np.abs(shifted_spike) ** 0.4)
                
                # High amplitude amplitude
                amp = state.uniform(1.2, 2.5)
                sample[:, ch] += amp * shifted_spike
                
        # Inject instrumentation thermal noise
        sample += state.normal(0, noise_level, (time_steps, n_channels))
        
        X.append(sample)
        y.append(label)
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    return X, y

def get_seizure_dataloader(batch_size=32, noise_level=0.2):
    X, y = generate_seizure_data(num_samples=500, noise_level=noise_level)
    
    # Split 400 train, 100 test
    X_train, X_test = X[:400], X[400:]
    y_train, y_test = y[:400], y[400:]
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
