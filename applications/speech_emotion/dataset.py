import os
import sys
import numpy as np
import librosa

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def load_real_ravdess(data_dir=None, use_deltas=True):
    """
    Loads RAVDESS speech recordings, extracts MFCCs, Delta, and Delta2.
    Pads/clips to length 200.
    """
    if data_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(project_root, "data", "datasets", "ravdess")
        
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ RAVDESS dataset directory not found at: {data_dir}")
        
    print(f"📖 Loading RAVDESS dataset from: {data_dir}")
    
    emotion_map = {'01': 0, '03': 1, '04': 2, '05': 3}
    X_raw, y = [], []
    actor_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    n_inputs = 39 if use_deltas else 13
    
    for actor_dir in actor_dirs:
        actor_path = os.path.join(data_dir, actor_dir)
        for file in sorted(os.listdir(actor_path)):
            if not file.endswith('.wav'):
                continue
            parts = file.split('-')
            emotion = parts[2]
            if emotion not in emotion_map:
                continue
            filepath = os.path.join(actor_path, file)
            try:
                signal, sr = librosa.load(filepath, sr=22050)
                mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
                if use_deltas:
                    delta = librosa.feature.delta(mfcc)
                    delta2 = librosa.feature.delta(mfcc, order=2)
                    feat = np.concatenate([mfcc, delta, delta2], axis=0)
                else:
                    feat = mfcc
                X_raw.append(feat.T)
                y.append(emotion_map[emotion])
            except Exception:
                continue
                
    max_len = min(max(len(x) for x in X_raw), 200)
    X = np.zeros((len(X_raw), max_len, n_inputs))
    for i, x in enumerate(X_raw):
        length = min(len(x), max_len)
        X[i, :length] = x[:length]
        
    print(f"✅ RAVDESS dataset loaded: {X.shape[0]} samples, shape: {X.shape}, labels distribution: {np.bincount(y)}")
    return X, np.array(y)
