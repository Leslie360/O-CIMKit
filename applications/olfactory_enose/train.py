import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from profiles.device_profile import DeviceProfile
from applications.olfactory_enose.dataset import generate_olfactory_enose_dataset
from applications.olfactory_enose.model import OlfactoryEnoseReservoir, OlfactoryQATClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_and_evaluate_enose(X_feat, y, out_classes, device_profile=None, hidden_dim=64, lr=0.005, epochs=100, batch_size=32, dropout_p=0.2):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    
    # Scale epochs by factor if running a quick benchmark evaluation
    resample_factor = 12 if epochs <= 5 else 1
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_feat, y)):
        X_train, X_test = X_feat[train_idx], X_feat[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.LongTensor(y_train).to(device)
        X_test_t = torch.FloatTensor(X_test).to(device)
        
        # Instantiate LSQ QAT MLP
        model = OlfactoryQATClassifier(
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
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model.train()
        for epoch in range(epochs):
            for _ in range(resample_factor):
                for bx, by in loader:
                    outputs = model(bx)
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
    parser = argparse.ArgumentParser(description="Olfactory Electronic Nose Gas Classification under CIM Constraints")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    
    if "ipykernel" in sys.modules or not sys.argv[0].endswith("train.py"):
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
        
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 1. Load Device Profiles
    nonvolatile_profile_path = os.path.join(project_root, "profiles", "repository", "FingerMemristor.json")
    nonvolatile_profile = DeviceProfile.from_json(nonvolatile_profile_path) if os.path.exists(nonvolatile_profile_path) else None
    
    print("=" * 60)
    print("CIM Platform - Olfactory Electronic Nose Gas Classification")
    print("=" * 60)
    if nonvolatile_profile:
        print(f"  Loaded Device Profile: {nonvolatile_profile.device_name}")
        print(f"  discrete states count: {nonvolatile_profile.discrete_states_count}")
    
    # 2. Generate E-Nose Dataset
    X_raw, y_raw = generate_olfactory_enose_dataset(num_samples=250, seq_len=80, num_sensors=8, num_gases=4)
    print(f"  Dataset generated: {X_raw.shape[0]} samples, shape: {X_raw.shape}, classes: 4")
    
    # 3. Create Volatile Reservoir
    enose_res = OlfactoryEnoseReservoir(
        taus=[1.5, 3.0, 6.0], 
        n_res=128, 
        n_inputs=8, 
        device=device
    )
    enose_res.to(device)
    enose_res.eval()
    
    # 4. Extract physical features
    print("  Extracting volatile reservoir time-series kinetic features...")
    features = []
    batch_size_feat = 64
    t0 = time.time()
    for i in range(0, len(X_raw), batch_size_feat):
        batch_X = torch.FloatTensor(X_raw[i:i+batch_size_feat]).to(device)
        with torch.no_grad():
            feat = enose_res(batch_X)
            features.append(feat.cpu().numpy())
    features = np.concatenate(features, axis=0)
    print(f"  Feature shape: {features.shape}, Extraction time: {time.time()-t0:.2f}s")
    
    # 5. Train & Evaluate with LSQ Quantization
    print("  Training LSQ QAT MLP classifier...")
    acc, f1, std = train_and_evaluate_enose(
        features, y_raw, 
        out_classes=4, 
        device_profile=nonvolatile_profile, 
        hidden_dim=64, 
        lr=args.lr, 
        epochs=args.epochs,
        batch_size=args.batch_size,
        dropout_p=0.2
    )
    print(f"🏆 Final Result: Accuracy = {acc:.2f}% ± {std:.2f}%, F1-Score = {f1:.4f}")

if __name__ == "__main__":
    main()
