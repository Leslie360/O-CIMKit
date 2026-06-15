import os
import sys
import time
import torch
import numpy as np
import argparse
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from profiles.device_profile import DeviceProfile
from core.layers import PhysicalReservoir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_ppg_mock_dataset(num_samples=100, seq_len=300):
    """
    Generates synthetic PPG and accelerometer sequences to simulate TROIKA dataset.
    Signals consist of: heart pulse component, respiratory component, and motion artifacts.
    """
    np.random.seed(42)
    t = np.linspace(0, 10, seq_len)
    
    X = []
    y = []
    
    for i in range(num_samples):
        # Target heart rate in beats per minute (BPM)
        bpm = np.random.uniform(60.0, 160.0)
        hr_freq = bpm / 60.0
        
        # PPG signal: basic heartbeat + respiratory modulation (0.2 Hz) + noise
        ppg = np.sin(2 * np.pi * hr_freq * t) + 0.3 * np.sin(2 * np.pi * 0.25 * t)
        
        # Accelerometer signal: motion artifacts matching step frequencies (e.g. running at 2 Hz)
        motion_freq = np.random.uniform(1.5, 3.0)
        acc_x = np.sin(2 * np.pi * motion_freq * t) + np.random.normal(0, 0.1, seq_len)
        acc_y = np.cos(2 * np.pi * motion_freq * t) + np.random.normal(0, 0.1, seq_len)
        
        # Combine PPG and Acc as inputs
        # Add motion artifact to PPG
        ppg_with_artifact = ppg + 0.8 * acc_x + np.random.normal(0, 0.05, seq_len)
        
        # Input features: shape (seq_len, 3) -> PPG, Acc_X, Acc_Y
        signal = np.stack([ppg_with_artifact, acc_x, acc_y], axis=1)
        X.append(signal)
        y.append(bpm)
        
    return np.array(X), np.array(y)

def run_ppg_reservoir_eval(profile, X, y, n_reservoir=1000, spectral_radius=0.8978, input_scale=0.9468, ridge_alpha=0.0097, leaking_rate=0.3079):
    # Setup physical reservoir
    # If profile is None, we use a default mock volatile device profile inside PhysicalReservoir
    if profile is None:
        # Create a mock float baseline profile (we can construct a default profile)
        profile = DeviceProfile(
            name="Ideal Float Reservoir",
            device_type="volatile",
            is_volatile=True,
            leaking_rate_volatile=0.3,
            conductance_min=0.0,
            conductance_max=1.0
        )
        
    reservoir = PhysicalReservoir(
        n_inputs=3,
        n_reservoir=n_reservoir,
        device_profile=profile,
        dual_scale=True,
        seed=42,
        device='cpu'
    )
    
    # Apply tuned leaking rate for hardware-aware run
    if profile is not None:
        reservoir.alpha_fast = leaking_rate
        
    # Scale input weights and reservoir weights
    if reservoir.dual_scale:
        reservoir.W_in_fast = reservoir.W_in_fast * input_scale
        reservoir.W_in_slow = reservoir.W_in_slow * input_scale
        reservoir.W_fast = reservoir._scale_spectral_radius(reservoir.W_fast, target=spectral_radius)
        reservoir.W_slow = reservoir._scale_spectral_radius(reservoir.W_slow, target=spectral_radius)
    else:
        reservoir.W_in = reservoir.W_in * input_scale
        reservoir.W = reservoir._scale_spectral_radius(reservoir.W, target=spectral_radius)
    
    print("  Extracting reservoir state representations...")
    states_list = []
    for seq in X:
        states = reservoir.process_sequence(seq) # shape (seq_len, n_reservoir)
        feat = reservoir.extract_temporal_features(states) # shape (5 * n_reservoir,)
        states_list.append(feat)
        
    states_arr = np.array(states_list)
    
    # Stratified split: 70% Train, 30% Test
    split_idx = int(0.7 * len(X))
    X_train, X_test = states_arr[:split_idx], states_arr[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Readout layer: Ridge Regression
    readout = Ridge(alpha=ridge_alpha)
    readout.fit(X_train, y_train)
    
    train_preds = readout.predict(X_train)
    test_preds = readout.predict(X_test)
    
    # NRMSE (Normalized Root Mean Squared Error)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_nrmse = (test_rmse / (np.max(y_test) - np.min(y_test))) * 100.0
    
    return test_nrmse, test_preds, y_test

def main():
    parser = argparse.ArgumentParser(description="PPG Heart Rate Estimation using Volatile Reservoir CIM")
    parser.add_argument("--reservoir-size", type=int, default=1000, help="Reservoir size")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs placeholder for benchmark compatibility")
    
    if "ipykernel" in sys.modules or not sys.argv[0].endswith("train.py"):
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
        
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    profile_path = os.path.join(project_root, "profiles", "repository", "Lorenz_Volatile.json")
    profile = DeviceProfile.from_json(profile_path) if os.path.exists(profile_path) else None
    
    print("=" * 60)
    print("CIM Platform - PPG Mobile Health Heart Rate Estimation")
    print("=" * 60)
    if profile:
        print(f"  Loaded Volatile Device Profile: {profile.device_name}")
        print(f"  Volatile relaxation leaking rate: {profile.leaking_rate_volatile:.4f}")
    else:
        print("  ⚠️ Volatile profile not found! Running default baseline.")
        
    X_ppg, y_bpm = get_ppg_mock_dataset()
    print(f"  Dataset size: {X_ppg.shape[0]} samples of sequence length {X_ppg.shape[1]}")
    
    t0 = time.time()
    # 1. Ideal software baseline
    print("📢 Phase 1: Running Pure Software Reservoir Baseline (Ideal Float)...")
    nrmse_float, _, _ = run_ppg_reservoir_eval(None, X_ppg, y_bpm, n_reservoir=args.reservoir_size)
    
    # 2. Hardware-aware simulation
    print("📢 Phase 2: Running Hardware-Aware Reservoir under Volatile Relaxation...")
    nrmse_hw, preds, targets = run_ppg_reservoir_eval(profile, X_ppg, y_bpm, n_reservoir=args.reservoir_size)
    
    duration = time.time() - t0
    
    print("\n  SIMULATION RESULTS & COMPARISON")
    print("=" * 60)
    print(f"  {'Metric':<25} | {'Ideal Float':<12} | {'Volatile HW':<16}")
    print("-" * 60)
    print(f"  {'Heart Rate NRMSE (%)':<25} | {nrmse_float:<12.4f} | {nrmse_hw:<16.4f}")
    print(f"  {'Estimation Gap (%)':<25} | {'N/A':<12} | {nrmse_hw - nrmse_float:<16.4f}")
    print("=" * 60)
    print(f"🏆 Final PPG Heart Rate Estimation NRMSE: {nrmse_hw:.4f}%")
    print(f"⏱️ Execution time: {duration:.2f}s")
    print("=" * 60)

if __name__ == "__main__":
    main()
