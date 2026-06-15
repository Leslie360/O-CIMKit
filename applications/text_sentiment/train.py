import os
import sys
import time
import torch
import numpy as np
import argparse
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from profiles.device_profile import DeviceProfile
from core.layers import PhysicalReservoir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_text_mock_dataset(num_samples=200, seq_len=40, embed_dim=64):
    """
    Generates synthetic review sentiment sequences.
    Positive reviews have embeddings clustered near a positive concept vector.
    Negative reviews have embeddings clustered near a negative concept vector.
    """
    np.random.seed(42)
    
    # Define concept embedding axes
    pos_axis = np.random.randn(embed_dim)
    pos_axis /= np.linalg.norm(pos_axis)
    
    neg_axis = np.random.randn(embed_dim)
    neg_axis /= np.linalg.norm(neg_axis)
    
    X = []
    y = []
    
    for i in range(num_samples):
        sentiment = np.random.randint(0, 2)
        y.append(sentiment)
        
        seq = []
        for t in range(seq_len):
            # Pos reviews have 70% positive words, neg reviews have 70% negative words
            if sentiment == 1:
                word_type = 1 if np.random.random() < 0.7 else 0
            else:
                word_type = 0 if np.random.random() < 0.7 else 1
                
            # Generate word vector
            if word_type == 1:
                # Positive word vector
                word_vec = pos_axis + np.random.normal(0, 0.4, embed_dim)
            else:
                # Negative word vector
                word_vec = neg_axis + np.random.normal(0, 0.4, embed_dim)
                
            word_vec /= np.linalg.norm(word_vec)
            seq.append(word_vec)
            
        X.append(np.array(seq))
        
    return np.array(X), np.array(y)

def run_sentiment_reservoir_eval(profile, X, y, n_reservoir=1000, spectral_radius=0.7767, input_scale=0.4000, ridge_alpha=59.7511, leaking_rate=0.3509):
    if profile is None:
        profile = DeviceProfile(
            name="Ideal Float Reservoir",
            device_type="volatile",
            is_volatile=True,
            leaking_rate_volatile=0.35,
            conductance_min=0.0,
            conductance_max=1.0
        )
        
    reservoir = PhysicalReservoir(
        n_inputs=X.shape[2],
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
    
    print("  Extracting text sequence reservoir states...")
    states_list = []
    for seq in X:
        states = reservoir.process_sequence(seq)
        feat = reservoir.extract_temporal_features(states)
        states_list.append(feat)
        
    states_arr = np.array(states_list)
    
    split_idx = int(0.75 * len(X))
    X_train, X_test = states_arr[:split_idx], states_arr[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Readout layer: Ridge Classifier
    clf = RidgeClassifier(alpha=ridge_alpha)
    clf.fit(X_train, y_train)
    
    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_preds) * 100.0
    test_acc = accuracy_score(y_test, test_preds) * 100.0
    
    return test_acc

def main():
    parser = argparse.ArgumentParser(description="Edge Text Sentiment Analysis using Volatile Reservoir CIM")
    parser.add_argument("--reservoir-size", type=int, default=800, help="Reservoir size")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs placeholder for benchmark compatibility")
    
    if "ipykernel" in sys.modules or not sys.argv[0].endswith("train.py"):
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
        
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    profile_path = os.path.join(project_root, "profiles", "repository", "Lorenz_Volatile.json")
    profile = DeviceProfile.from_json(profile_path) if os.path.exists(profile_path) else None
    
    print("=" * 60)
    print("CIM Platform - Edge Text Sentiment Analysis (IMDB-like)")
    print("=" * 60)
    if profile:
        print(f"  Loaded Volatile Device Profile: {profile.device_name}")
        print(f"  Volatile relaxation leaking rate: {profile.leaking_rate_volatile:.4f}")
    else:
        print("  ⚠️ Volatile profile not found! Running default baseline.")
        
    X_text, y_sent = get_text_mock_dataset()
    print(f"  Dataset size: {X_text.shape[0]} samples of sequence length {X_text.shape[1]}")
    
    t0 = time.time()
    # 1. Ideal float
    print("📢 Phase 1: Running Pure Software Reservoir Baseline (Ideal Float)...")
    acc_float = run_sentiment_reservoir_eval(None, X_text, y_sent, n_reservoir=args.reservoir_size)
    
    # 2. Volatile hardware-aware
    print("📢 Phase 2: Running Hardware-Aware Reservoir under Volatile Relaxation...")
    acc_hw = run_sentiment_reservoir_eval(profile, X_text, y_sent, n_reservoir=args.reservoir_size)
    
    duration = time.time() - t0
    
    print("\n  SIMULATION RESULTS & COMPARISON")
    print("=" * 60)
    print(f"  {'Metric':<25} | {'Ideal Float':<12} | {'Volatile HW':<16}")
    print("-" * 60)
    print(f"  {'Sentiment Acc (%)':<25} | {acc_float:<12.2f} | {acc_hw:<16.2f}")
    print(f"  {'Accuracy Gap (%)':<25} | {'N/A':<12} | {acc_float - acc_hw:<16.2f}")
    print("=" * 60)
    print(f"🏆 Final Sentiment Accuracy: {acc_hw:.2f}%")
    print(f"⏱️ Execution time: {duration:.2f}s")
    print("=" * 60)

if __name__ == "__main__":
    main()
