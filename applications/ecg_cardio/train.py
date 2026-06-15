import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from profiles.device_profile import DeviceProfile
from applications.ecg_cardio.dataset import load_ecg_data
from applications.ecg_cardio.model import OptoelectronicReservoir, ECGQATClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_and_evaluate_ecg(X_feat, y, out_classes, device_profile=None, hidden_dim=256, lr=0.005, epochs=150, dropout_p=0.3):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_feat, y)):
        X_train, X_test = X_feat[train_idx], X_feat[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.LongTensor(y_train).to(device)
        X_test_t = torch.FloatTensor(X_test).to(device)
        
        # Instantiate LSQ QAT MLP with Device Profile
        model = ECGQATClassifier(
            in_features=X_feat.shape[1], 
            hidden_dim=hidden_dim, 
            out_classes=out_classes, 
            device_profile=device_profile, 
            dropout_p=dropout_p
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        model.train()
        for epoch in range(epochs):
            for bx, by in loader:
                # Add 5% input noise for physical noise regularization
                noise = torch.randn_like(bx) * 0.05
                outputs = model(bx + noise)
                loss = criterion(outputs, by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_t)
            preds = outputs.argmax(dim=-1).cpu().numpy()
            
        acc = accuracy_score(y_test, preds) * 100
        f1 = f1_score(y_test, preds, average='macro')
        accs.append(acc)
        f1s.append(f1)
        print(f"  Fold {fold+1}/5 - Accuracy: {acc:.2f}%, F1-Score: {f1:.4f}")
        
    return np.mean(accs), np.mean(f1s), np.std(accs)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ECG Arrhythmia Classification under CIM Constraints")
    parser.add_argument("--dataset", type=str, default="mitdb", choices=["mitdb", "ptbdb"], help="ECG dataset to use")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    
    if "ipykernel" in sys.modules or not sys.argv[0].endswith("train.py"):
        args = parser.parse_args([])
        if not hasattr(args, "dataset"):
            args.dataset = "mitdb"
        if not hasattr(args, "epochs"):
            args.epochs = 150
    else:
        args = parser.parse_args()
        
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 1. Load Device Profiles
    volatile_profile_path = os.path.join(project_root, "profiles", "repository", "PFO_AlOx_Volatile.json")
    nonvolatile_profile_path = os.path.join(project_root, "profiles", "repository", "PFO_AlOx_NonVolatile.json")
    
    volatile_profile = DeviceProfile.from_json(volatile_profile_path) if os.path.exists(volatile_profile_path) else None
    nonvolatile_profile = DeviceProfile.from_json(nonvolatile_profile_path) if os.path.exists(nonvolatile_profile_path) else None
    
    print("=" * 60)
    print(f"CIM Platform - ECG Arrhythmia Classification ({args.dataset.upper()})")
    print("=" * 60)
    
    # 2. Load ECG Dataset
    X_ecg, y_ecg = load_ecg_data(dataset_name=args.dataset)
    print(f"  Input size: {X_ecg.shape}, Abnormal ratio: {np.mean(y_ecg)*100:.2f}%")
    
    # 3. Create optoelectronic physical reservoir
    ecg_res = OptoelectronicReservoir(device_profile=volatile_profile, n_inputs=1, n_reservoir=2000, device=device)
    ecg_res.to(device)
    ecg_res.eval()
    
    # 4. Extract physical features
    print("  Extracting optoelectronic reservoir features...")
    ecg_features = []
    batch_size = 256
    t0 = time.time()
    for i in range(0, len(X_ecg), batch_size):
        batch_X = torch.FloatTensor(X_ecg[i:i+batch_size]).to(device)
        with torch.no_grad():
            states = ecg_res(batch_X)
            feat = ecg_res.extract_features(states)
            ecg_features.append(feat.cpu().numpy())
    ecg_features = np.concatenate(ecg_features, axis=0)
    print(f"  Feature shape: {ecg_features.shape}, Extraction time: {time.time()-t0:.2f}s")
    
    # 5. Train & Evaluate Float Baseline
    print("📢 Phase 1: Training Pure Software Float Baseline (Ideal Float)...")
    acc_float, f1_float, std_float = train_and_evaluate_ecg(
        ecg_features, y_ecg, 
        out_classes=2, 
        device_profile=None, 
        hidden_dim=256, 
        lr=0.005, 
        epochs=args.epochs, 
        dropout_p=0.3
    )
    
    # 6. Train & Evaluate with LSQ Quantization
    print("📢 Phase 2: Training Hardware-Aware Model under Memristive Constraints...")
    acc_hw, f1_hw, std_hw = train_and_evaluate_ecg(
        ecg_features, y_ecg, 
        out_classes=2, 
        device_profile=nonvolatile_profile, 
        hidden_dim=256, 
        lr=0.005, 
        epochs=args.epochs, 
        dropout_p=0.3
    )
    
    # Report results
    print("  SIMULATION RESULTS & COMPARISON")
    print("=" * 60)
    print(f"  {'Metric':<25} | {'Ideal Float':<12} | {'Memristive HW':<16}")
    print("-" * 60)
    print(f"  {'Classification Acc (%)':<25} | {acc_float:<12.2f} | {acc_hw:<16.2f}")
    print(f"  {'F1-Score':<25} | {f1_float:<12.4f} | {f1_hw:<16.4f}")
    print(f"  {'Accuracy Loss Gap (%)':<25} | {'N/A':<12} | {acc_float - acc_hw:<16.2f}")
    print("=" * 60)
    
    print(f"🏆 Final Result: Accuracy = {acc_hw:.2f}% ± {std_hw:.2f}%, F1-Score = {f1_hw:.4f}")

if __name__ == "__main__":
    main()
