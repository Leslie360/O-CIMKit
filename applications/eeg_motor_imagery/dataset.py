import numpy as np

def generate_eeg_motor_imagery_dataset(num_samples=240, seq_len=100, num_channels=16, num_classes=4):
    """
    Generates synthetic EEG motor imagery trials (16 channels, 4 classes).
    Classes: 0 (Left Hand), 1 (Right Hand), 2 (Feet), 3 (Rest).
    Simulates Event-Related Desynchronization (ERD) in Mu (8-12 Hz) and Beta (13-30 Hz) bands:
    - Left Hand: Mu desynchronization in Right Motor Cortex (channels 10-12, e.g. C4-like).
    - Right Hand: Mu desynchronization in Left Motor Cortex (channels 4-6, e.g. C3-like).
    - Feet: Mu desynchronization in Central Motor Cortex (channels 7-9, e.g. Cz-like).
    - Rest: Normal rhythm (8-12 Hz and 13-30 Hz oscillations) across all channels.
    """
    np.random.seed(42)
    t = np.linspace(0, 4, seq_len) # 4 seconds trials
    
    X = []
    y = []
    
    # Base frequencies
    freq_mu = 10.0  # 10 Hz Mu rhythm
    freq_beta = 22.0 # 22 Hz Beta rhythm
    
    for i in range(num_samples):
        cls_id = np.random.randint(0, num_classes)
        y.append(cls_id)
        
        trial_signals = []
        for ch in range(num_channels):
            # Base noise
            noise = np.random.normal(0, 0.4, seq_len)
            
            # Oscillations
            mu_wave = np.sin(2 * np.pi * freq_mu * t)
            beta_wave = np.sin(2 * np.pi * freq_beta * t)
            
            # Apply class-specific desynchronization (ERD)
            amp_scale = 1.0
            
            if cls_id == 0: # Left Hand (Right hemisphere ERD)
                if 10 <= ch <= 12:
                    # ERD: amplitude decays exponentially after t=1.0s
                    erd_envelope = np.ones(seq_len)
                    decay_start = seq_len // 4
                    erd_envelope[decay_start:] = np.exp(-t[decay_start:] * 0.8)
                    amp_scale = 0.25 * erd_envelope
            elif cls_id == 1: # Right Hand (Left hemisphere ERD)
                if 4 <= ch <= 6:
                    erd_envelope = np.ones(seq_len)
                    decay_start = seq_len // 4
                    erd_envelope[decay_start:] = np.exp(-t[decay_start:] * 0.8)
                    amp_scale = 0.25 * erd_envelope
            elif cls_id == 2: # Feet (Central hemisphere ERD)
                if 7 <= ch <= 9:
                    erd_envelope = np.ones(seq_len)
                    decay_start = seq_len // 4
                    erd_envelope[decay_start:] = np.exp(-t[decay_start:] * 0.8)
                    amp_scale = 0.25 * erd_envelope
            # Rest (cls_id == 3) keeps amp_scale = 1.0 (no ERD)
            
            signal = (mu_wave + 0.4 * beta_wave) * amp_scale + noise
            trial_signals.append(signal)
            
        # Shape: (seq_len, num_channels)
        X.append(np.stack(trial_signals, axis=1))
        
    return np.array(X), np.array(y)
