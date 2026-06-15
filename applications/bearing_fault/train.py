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
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from profiles.device_profile import DeviceProfile
from applications.bearing_fault.dataset import load_cwru_data
from applications.bearing_fault.model import BearingMultiScaleReservoir, BearingQATClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_and_evaluate_qat(X_feat, y, out_classes, device_profile=None, hidden_dim=128, lr=0.002, epochs=100, dropout_p=0.3):
    """Evaluate under 64-state hardware QAT MLP."""
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
        
        model = BearingQATClassifier(
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
            
        accs.append(accuracy_score(y_test, preds) * 100)
        f1s.append(f1_score(y_test, preds, average='macro'))
        
    return np.mean(accs), np.mean(f1s), np.std(accs)

def train_classical_classifiers(X_feat, y):
    """Evaluate classical digital classifiers (Ridge, SVM, RF)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    classifiers = {
        'DualMode_RC_Ridge': lambda: RidgeClassifier(alpha=1.0),
        'DualMode_RC_SVM': lambda: SVC(kernel='rbf', C=10, random_state=42),
        'RF': lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    }
    
    results = {}
    for name, clf_factory in classifiers.items():
        accs, f1s = [], []
        for train_idx, test_idx in skf.split(X_feat, y):
            X_train, X_test = X_feat[train_idx], X_feat[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            clf = clf_factory()
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            
            accs.append(accuracy_score(y_test, preds) * 100)
            f1s.append(f1_score(y_test, preds, average='macro'))
            
        results[name] = {
            'acc': np.mean(accs), 'acc_std': np.std(accs),
            'f1': np.mean(f1s), 'f1_std': np.std(f1s)
        }
        print(f"  {name}: Acc = {results[name]['acc']:.2f}% ± {results[name]['acc_std']:.2f}%, F1 = {results[name]['f1']:.4f}")
        
    return results

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 1. Load Device Profiles
    nonvolatile_profile_path = os.path.join(project_root, "profiles", "repository", "PFO_AlOx_NonVolatile.json")
    nonvolatile_profile = DeviceProfile.from_json(nonvolatile_profile_path) if os.path.exists(nonvolatile_profile_path) else None
    
    print("=" * 60)
    print("CIM Platform - CWRU Bearing Fault Classification")
    print("=" * 60)
    
    # 2. Load CWRU Dataset
    X_cwru, y_cwru = load_cwru_data()
    
    # 3. Create Multi-Scale Reservoir
    bearing_res = BearingMultiScaleReservoir(taus=[3, 10, 30], n_reservoir=500, n_inputs=1, device=device)
    
    # 4. Extract physical features
    print("  Extracting multi-scale reservoir features...")
    t0 = time.time()
    bearing_features = []
    batch_size = 256
    for i in range(0, len(X_cwru), batch_size):
        batch_X = torch.FloatTensor(X_cwru[i:i+batch_size]).to(device)
        with torch.no_grad():
            feat = bearing_res(batch_X)
            bearing_features.append(feat.cpu().numpy())
    bearing_features = np.concatenate(bearing_features, axis=0)
    print(f"  Feature shape: {bearing_features.shape}, Extraction time: {time.time()-t0:.2f}s")
    
    # 5. Evaluate Classical Classifiers
    print("  Evaluating Classical Classifiers...")
    train_classical_classifiers(bearing_features, y_cwru)
    
    # 6. Evaluate Float Baseline
    print("📢 Phase 1: Training Pure Software Float Baseline (Ideal Float)...")
    acc_float, f1_float, std_float = train_and_evaluate_qat(
        bearing_features, y_cwru, 
        out_classes=len(set(y_cwru)), 
        device_profile=None, 
        hidden_dim=128, 
        lr=0.002, 
        epochs=100, 
        dropout_p=0.3
    )
    
    # 7. Evaluate LSQ QAT MLP
    print("📢 Phase 2: Training Hardware-Aware Model under Memristive Constraints...")
    acc_hw, f1_hw, std_hw = train_and_evaluate_qat(
        bearing_features, y_cwru, 
        out_classes=len(set(y_cwru)), 
        device_profile=nonvolatile_profile, 
        hidden_dim=128, 
        lr=0.002, 
        epochs=100, 
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
    
    print(f"🏆 LSQ QAT MLP: Accuracy = {acc_hw:.2f}% ± {std_hw:.2f}%, F1-Score = {f1_hw:.4f}")

if __name__ == "__main__":
    main()
