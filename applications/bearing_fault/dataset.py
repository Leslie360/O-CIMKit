import os
import sys
import numpy as np
from scipy.io import loadmat

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def load_cwru_data(data_dir=None):
    """
    Loads DE (Drive End accelerometer) vibration data from case western bearing dataset.
    Returns segmented signal sections and labels.
    """
    if data_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(project_root, "data", "datasets", "cwru")
        
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ CWRU dataset directory not found at: {data_dir}")
        
    print(f"📖 Loading CWRU bearing data from: {data_dir}")
    
    # We expect files normal.mat, fault_7.mat, fault_14.mat, fault_21.mat
    files = {
        'normal': 'normal.mat',
        'fault_7': 'fault_7.mat',
        'fault_14': 'fault_14.mat',
        'fault_21': 'fault_21.mat'
    }
    
    X_list, y_list = [], []
    label = 0
    
    for name, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"⚠️ Warning: File {filename} not found. Synthesizing data fallback...")
            return generate_synthetic_bearing_data()
            
        try:
            mat = loadmat(filepath)
            for key in mat.keys():
                if 'DE' in key:  # Drive End vibration readings
                    data = mat[key].flatten()
                    seg_len = 500
                    n_segs = min(len(data) // seg_len, 250)
                    for i in range(n_segs):
                        seg = data[i*seg_len:(i+1)*seg_len]
                        X_list.append(seg)
                        y_list.append(label)
            label += 1
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}. Fallback to synthetic...")
            return generate_synthetic_bearing_data()
            
    X = np.array(X_list).reshape(-1, 500, 1)
    y = np.array(y_list)
    
    print(f"✅ CWRU dataset loaded: {X.shape[0]} samples, shape: {X.shape}, labels distribution: {np.bincount(y)}")
    return X, y

def generate_synthetic_bearing_data():
    """Fallback generator for synthetic vibration data."""
    print("生成合成轴承振动数据...")
    np.random.seed(42)
    n_samples = 1000
    seq_len = 500
    n_classes = 4
    
    X = np.zeros((n_samples, seq_len, 1))
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        fault_type = i % n_classes
        y[i] = fault_type
        t = np.linspace(0, 1, seq_len)
        base_freq = 30
        
        if fault_type == 0:
            signal = np.sin(2 * np.pi * base_freq * t) + 0.1 * np.random.randn(seq_len)
        elif fault_type == 1:
            signal = np.sin(2 * np.pi * base_freq * t) + 0.2 * np.random.randn(seq_len) + 0.3 * np.sin(2 * np.pi * 100 * t)
        elif fault_type == 2:
            signal = np.sin(2 * np.pi * base_freq * t) + 0.3 * np.random.randn(seq_len) + 0.5 * np.sin(2 * np.pi * 100 * t) + 0.2 * np.sin(2 * np.pi * 200 * t)
        else:
            signal = np.sin(2 * np.pi * base_freq * t) + 0.4 * np.random.randn(seq_len) + 0.7 * np.sin(2 * np.pi * 100 * t) + 0.4 * np.sin(2 * np.pi * 200 * t) + 0.3 * np.sin(2 * np.pi * 300 * t)
            
        X[i, :, 0] = signal
        
    return X, y
