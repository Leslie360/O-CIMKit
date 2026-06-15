import numpy as np

def generate_neuromorphic_kws_dataset(num_samples=300, seq_len=50, num_features=39, num_classes=4):
    """
    Generates synthetic Keyword Spotting (KWS) features.
    Features: 39-dimensional MFCC frames.
    Classes: 0 ("Yes"), 1 ("No"), 2 ("Go"), 3 (Silence/Background Noise).
    Simulates spectral-temporal signatures of spoken words:
    - "Yes" (Class 0): Peak in high-frequency coefficients (index 20-30) in first half.
    - "No" (Class 1): Peak in low-frequency coefficients (index 1-10) in middle.
    - "Go" (Class 2): High energy burst across all coefficients at the end.
    - Silence/Noise (Class 3): Stationary brownian noise with no structured peaks.
    """
    np.random.seed(100)
    
    X = []
    y = []
    
    for i in range(num_samples):
        cls_id = np.random.randint(0, num_classes)
        y.append(cls_id)
        
        # Base background noise
        mfcc_frames = np.random.normal(0, 0.2, (seq_len, num_features))
        
        if cls_id == 0:  # "Yes"
            # High-freq energy envelope starting early
            for t in range(5, 20):
                mfcc_frames[t, 20:30] += 1.5 * np.sin(np.pi * (t - 5) / 15)
        elif cls_id == 1:  # "No"
            # Low-freq energy envelope centered
            for t in range(15, 35):
                mfcc_frames[t, 1:10] += 1.6 * np.sin(np.pi * (t - 15) / 20)
        elif cls_id == 2:  # "Go"
            # Brief explosive burst at the end
            for t in range(35, 45):
                mfcc_frames[t, :] += 1.8 * np.sin(np.pi * (t - 35) / 10)
        # Class 3 has no added word envelope
        
        X.append(mfcc_frames)
        
    return np.array(X), np.array(y)
