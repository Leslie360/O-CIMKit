import os
import sys
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from profiles.device_profile import DeviceProfile
from applications.edge_llm.dataset import load_mitbih_data
from applications.edge_llm.model import EdgeLLMSentinel

# Energy constants
ENERGY_PER_OP = 10e-15  # 10 fJ
ENERGY_LLM_TOKEN = 1e-6  # 1 uJ

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Edge-LLM Sentinel Anomaly Interceptor")
    parser.add_argument("--epochs", type=int, default=3, help="Dummy epochs for benchmark compatibility")
    
    if "ipykernel" in sys.modules or not sys.argv[0].endswith("train.py"):
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
        
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 1. Load Device Profiles
    volatile_profile_path = os.path.join(project_root, "profiles", "repository", "PFO_AlOx_Volatile.json")
    volatile_profile = DeviceProfile.from_json(volatile_profile_path) if os.path.exists(volatile_profile_path) else None
    
    nonvolatile_profile_path = os.path.join(project_root, "profiles", "repository", "PFO_AlOx_NonVolatile.json")
    nonvolatile_profile = DeviceProfile.from_json(nonvolatile_profile_path) if os.path.exists(nonvolatile_profile_path) else None
    
    tau_val = volatile_profile.tau_volatile if volatile_profile else 3.64
    n_states = nonvolatile_profile.discrete_states_count if nonvolatile_profile else 64
    
    print("=" * 60)
    print("CIM Platform - Edge-LLM Sentinel Anomaly Interceptor")
    print("=" * 60)
    print(f"  Volatile Device Tau: {tau_val}s")
    print(f"  Non-volatile Device States: {n_states}")

    # 2. Load Dataset
    normal_beats, anomaly_beats = load_mitbih_data()
    
    # Create Sentinel
    sentinel = EdgeLLMSentinel(device_profile=volatile_profile, tau=tau_val, n_states=n_states)
    
    # 3. Train & Evaluate
    np.random.seed(42)
    n_train_normal = min(1000, len(normal_beats) // 2)
    n_train_anomaly = min(500, len(anomaly_beats) // 2)
    n_test_normal = min(2000, len(normal_beats) - n_train_normal)
    n_test_anomaly = min(500, len(anomaly_beats) - n_train_anomaly)

    train_normal = normal_beats[:n_train_normal]
    train_anomaly = anomaly_beats[:n_train_anomaly]
    test_normal = normal_beats[n_train_normal:n_train_normal + n_test_normal]
    test_anomaly = anomaly_beats[n_train_anomaly:n_train_anomaly + n_test_anomaly]

    print(f"  Training set size: Normal = {len(train_normal)}, Anomaly = {len(train_anomaly)}")
    print(f"  Testing set size:  Normal = {len(test_normal)}, Anomaly = {len(test_anomaly)}")

    # Train Sentinel
    sentinel.train(train_normal, train_anomaly)

    # Evaluate on test set
    print("\n  Evaluating Sentinel interception performance...")
    normal_results = []
    for beat in test_normal:
        is_anomaly, prob = sentinel.detect(beat)
        normal_results.append((is_anomaly, prob))

    anomaly_results = []
    for beat in test_anomaly:
        is_anomaly, prob = sentinel.detect(beat)
        anomaly_results.append((is_anomaly, prob))

    normal_results = np.array(normal_results)
    anomaly_results = np.array(anomaly_results)

    # Calculate metrics
    interception_rate = np.mean(~normal_results[:, 0].astype(bool))
    detection_rate = np.mean(anomaly_results[:, 0].astype(bool))

    # Energy reduction calculation
    n_total = len(test_normal) + len(test_anomaly)
    n_wakeup_normal = np.sum(normal_results[:, 0].astype(bool))
    n_wakeup_anomaly = np.sum(anomaly_results[:, 0].astype(bool))
    n_wakeup = n_wakeup_normal + n_wakeup_anomaly
    n_intercepted = n_total - n_wakeup

    energy_intercepted = n_intercepted * ENERGY_PER_OP
    # Assume 100 tokens needed for LLM diagnosis when woken up
    energy_wakeup = n_wakeup * (ENERGY_PER_OP + ENERGY_LLM_TOKEN * 100)
    energy_total = energy_intercepted + energy_wakeup

    energy_baseline = n_total * (ENERGY_PER_OP + ENERGY_LLM_TOKEN * 100)
    energy_reduction = energy_baseline / energy_total

    # Classification Metrics
    y_true = np.concatenate([np.zeros(len(test_normal)), np.ones(len(test_anomaly))])
    y_pred = np.concatenate([normal_results[:, 0], anomaly_results[:, 0]]).astype(int)
    y_prob = np.concatenate([normal_results[:, 1], anomaly_results[:, 1]])

    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    print("-" * 60)
    print(f"🏆 Interception Rate (Normal beats): {interception_rate:.2%}")
    print(f"🏆 Anomaly Detection Rate:           {detection_rate:.2%}")
    print(f"🏆 System Energy Reduction:          {energy_reduction:.1f}x")
    print(f"🏆 F1-Score:                         {f1:.4f}")
    print(f"🏆 AUC ROC:                          {auc:.4f}")
    print("-" * 60)

if __name__ == "__main__":
    main()
