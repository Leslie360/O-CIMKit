import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from profiles.device_profile import DeviceProfile
from core.layers import QATMLPLayer
from applications.embodied_ai.dataset import generate_tactile_dataset_with_temporal_dependency
from applications.embodied_ai.model import EmbodiedAIRC

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EmbodiedQATClassifier(nn.Module):
    def __init__(self, in_features, out_classes, device_profile=None):
        super().__init__()
        self.fc1 = QATMLPLayer(in_features, 128, device_profile=device_profile)
        self.fc2 = QATMLPLayer(128, 64, device_profile=device_profile)
        self.fc3 = QATMLPLayer(64, 32, device_profile=device_profile)
        self.fc4 = QATMLPLayer(32, out_classes, device_profile=device_profile)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        h = self.dropout(self.relu(self.fc1(x)))
        h = self.dropout(self.relu(self.fc2(h)))
        h = self.relu(self.fc3(h))
        return self.fc4(h)

def train_and_evaluate(X, y, rc_model, material_baselines, device_profile=None, n_splits=5, epochs=20, batch_size=32):
    n_samples = X.shape[0]
    features = []

    for i in range(n_samples):
        states = rc_model.process_tactile_signal(X[i], material_baselines[i])
        feat = [
            np.mean(states[:, 0]), np.std(states[:, 0]), np.max(states[:, 0]),
            np.min(states[:, 0]), np.percentile(states[:, 0], 25),
            np.percentile(states[:, 0], 75),
            np.mean(states[:, 1]), np.std(states[:, 1])
        ]
        features.append(feat)

    features = np.array(features)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(features, y)):
        X_train = torch.FloatTensor(features[train_idx]).to(device)
        y_train = torch.LongTensor(y[train_idx]).to(device)
        X_test = torch.FloatTensor(features[test_idx]).to(device)
        y_test = torch.LongTensor(y[test_idx]).to(device)

        model = EmbodiedQATClassifier(in_features=features.shape[1], out_classes=4, device_profile=device_profile).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in train_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            preds = outputs.argmax(dim=-1)
            acc = accuracy_score(y_test.cpu(), preds.cpu())
            f1 = f1_score(y_test.cpu(), preds.cpu(), average='weighted')
            accs.append(acc)
            f1s.append(f1)

    return np.mean(accs), np.std(accs), np.mean(f1s)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Embodied AI Multimodality Sensor Fusion")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    
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
    print("CIM Platform - Embodied AI Multimodality Sensor Fusion")
    print("=" * 60)
    print(f"  Volatile Device Tau: {tau_val}s")
    print(f"  Non-volatile Device States: {n_states}")

    # 2. Generate tactile dataset
    print("  Generating tactile dataset with temporal dependency...")
    X, y, materials, material_baselines = generate_tactile_dataset_with_temporal_dependency(
        n_samples=2000, n_timesteps=100, tau_volatile=tau_val
    )
    print(f"  Tactile data generated: {X.shape[0]} samples, shape: {X.shape}")

    # 3. Experiment 1: Timescale alignment sweep
    print("\n  Experiment 1: Scanning volatile timescales...")
    taus = [0.1, 0.5, 1.0, 2.0, tau_val, 5.0, 10.0, 30.0]
    results = []
    
    print("-" * 60)
    print(f"{'tau (s)':>10} {'Accuracy':>12} {'F1-score':>12} {'Is Device?'}")
    print("-" * 60)
    
    for t_scan in taus:
        # Create temp profile with scanned tau
        temp_profile = DeviceProfile(
            name="TempScan", 
            tau_volatile=t_scan, 
            is_volatile=True,
            discrete_states_count=n_states
        )
        rc = EmbodiedAIRC(device_profile=temp_profile)
        acc_mean, acc_std, f1 = train_and_evaluate(X, y, rc, material_baselines, device_profile=nonvolatile_profile, epochs=args.epochs)
        results.append((t_scan, acc_mean, acc_std, f1))
        
        is_device_flag = "<- YES" if abs(t_scan - tau_val) < 0.01 else ""
        print(f"{t_scan:>10.2f} {acc_mean:>12.2%} {f1:>12.3f} {is_device_flag}")
        
    print("-" * 60)
    best_tau, best_acc, best_std, best_f1 = max(results, key=lambda x: x[1])
    print(f"🏆 Best Timescale: {best_tau:.2f}s with Accuracy: {best_acc:.2%}")

    # 4. Experiment 2: Energy calculation comparison
    print("\n  Experiment 2: Energy comparison (Physics vs GPU baseline)")
    n_samples = len(X)
    n_timesteps = X.shape[1]
    
    energy_per_op = 10e-15  # 10 fJ
    energy_gpu = 1e-9      # 1 nJ
    downsample_factor = 10
    
    energy_rc = n_samples * n_timesteps * energy_per_op
    energy_gpu_total = n_samples * (n_timesteps // downsample_factor) * energy_gpu
    energy_ratio = energy_gpu_total / energy_rc
    
    print(f"  Physical RC Energy consumption: {energy_rc*1e12:.2f} pJ")
    print(f"  Digital GPU Energy consumption: {energy_gpu_total*1e12:.2f} pJ")
    print(f"  🏆 Energy Efficiency Gain: {energy_ratio:.0f}x")

if __name__ == "__main__":
    main()
