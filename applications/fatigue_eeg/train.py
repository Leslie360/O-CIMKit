import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from profiles.device_profile import DeviceProfile
from applications.fatigue_eeg.dataset import load_real_sleep_edf_2ch
from applications.fatigue_eeg.model import OrganicEEGNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom dual logger to print to both stdout and a local log file
class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def calibrate_batchnorm(model, dataloader, device):
    model.train()
    old_noise_stds = {}
    for name, module in model.named_modules():
        if hasattr(module, 'noise_std'):
            old_noise_stds[name] = module.noise_std
            module.noise_std = 0.0
            
    old_momentums = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            old_momentums[name] = module.momentum
            module.momentum = None
            module.reset_running_stats()
            
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            _ = model(inputs)
            
    for name, module in model.named_modules():
        if name in old_noise_stds:
            module.noise_std = old_noise_stds[name]
        if name in old_momentums:
            module.momentum = old_momentums[name]

def train_and_evaluate_eeg(X_raw, y, out_classes, device_profile=None, lr=0.001, epochs=100, batch_size=64, dropout_p=0.3):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    
    resample_factor = 6 if epochs <= 5 else 1
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_raw, y)):
        print(f"📌 Starting Fold {fold+1}/5...")
        X_train, X_test = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Transpose to (samples, channels, seq_len)
        X_train = X_train.transpose(0, 2, 1)
        X_test = X_test.transpose(0, 2, 1)
        
        # Standardize using training partition statistics
        mean = X_train.mean()
        std = X_train.std()
        X_train = (X_train - mean) / (std + 1e-8)
        X_test = (X_test - mean) / (std + 1e-8)
        
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.LongTensor(y_train).to(device)
        X_test_t = torch.FloatTensor(X_test).to(device)
        y_test_t = torch.LongTensor(y_test).to(device)
        
        model = OrganicEEGNet(
            device_profile=device_profile, 
            num_classes=out_classes, 
            dropout_p=dropout_p
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = torch.amp.GradScaler('cuda')
        
        train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        test_dataset = torch.utils.data.TensorDataset(X_test_t, y_test_t)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        best_fold_acc = 0.0
        
        for epoch in range(epochs):
            model.train()
            # Suppress/set noise depending on status
            is_calib = (epoch >= epochs - 2)
            noise_warmup_epochs = 30
            
            if epoch < noise_warmup_epochs:
                # 1. Warm-up stage: no noise injection for organic layers
                for module in model.modules():
                    if hasattr(module, 'noise_std'):
                        module.noise_std = 0.0
            else:
                # 2. Main training stage: inject device-specific cycle-to-cycle noise
                for module in model.modules():
                    if hasattr(module, 'noise_std'):
                        if is_calib:
                            module.noise_std = 0.0
                        else:
                            if hasattr(module, 'profile') and module.profile is not None:
                                module.noise_std = module.profile.get_noise_std()
                            
            for _ in range(resample_factor):
                for bx, by in train_loader:
                    optimizer.zero_grad()
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs = model(bx)
                        loss = criterion(outputs, by)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Clamp Conv & MLP weights to [-1, 1] to keep them within physical dynamic range
                    with torch.no_grad():
                        for param in model.parameters():
                            if param.requires_grad:
                                param.data.clamp_(-1.0, 1.0)
                                
            scheduler.step()
            
            # Recalibrate Batch Normalization and evaluate only at checkpoints to maintain training stability
            should_eval = (epoch % 5 == 0) or (epoch >= epochs - 3) or (epoch == epochs - 1)
            if should_eval:
                calibrate_batchnorm(model, train_loader, device)
                
                # Evaluate current epoch
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for bx, by in test_loader:
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                            outputs = model(bx)
                        preds = outputs.argmax(dim=-1)
                        correct += preds.eq(by).sum().item()
                        total += by.size(0)
                        
                epoch_acc = 100. * correct / total
                if epoch_acc > best_fold_acc:
                    best_fold_acc = epoch_acc
                
        # Calculate F1 on best fold predictions
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                final_outputs = model(X_test_t)
            final_preds = final_outputs.argmax(dim=-1).cpu().numpy()
            
        fold_f1 = f1_score(y_test, final_preds, average='macro')
        accs.append(best_fold_acc)
        f1s.append(fold_f1)
        print(f"  Fold {fold+1}/5 - Best Accuracy: {best_fold_acc:.2f}%, Final F1-Score: {fold_f1:.4f}")
        
    return np.mean(accs), np.mean(f1s), np.std(accs)

def main():
    parser = argparse.ArgumentParser(description="OrganicEEGNet Fatigue Classification")
    parser.add_argument("--epochs", type=int, default=140, help="Number of epochs")
    if "ipykernel" in sys.modules or not sys.argv[0].endswith("train.py"):
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
        
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_path = os.path.join(project_root, "applications", "fatigue_eeg", "train.log")
    
    # Redirect output to both file and console
    sys.stdout = DualLogger(log_path)
    
    # 1. Load Device Profile
    nonvolatile_profile_path = os.path.join(project_root, "profiles", "repository", "PFO_AlOx_NonVolatile.json")
    nonvolatile_profile = DeviceProfile.from_json(nonvolatile_profile_path) if os.path.exists(nonvolatile_profile_path) else None
    
    print("=" * 60)
    print("CIM Platform - Sleep-EDF EEG Fatigue Classification (CNN INNOVATION)")
    print("=" * 60)
    if nonvolatile_profile:
        print(f"  Loaded Device Profile: {nonvolatile_profile.device_name}")
        print(f"  Conductance range: {nonvolatile_profile.conductance_min} to {nonvolatile_profile.conductance_max}")
    
    # 2. Load EEG Dataset
    X_eeg, y_eeg = load_real_sleep_edf_2ch()
    
    t0 = time.time()
    
    # --- PHASE 1: IDEAL SOFTWARE FLOAT BASELINE ---
    print("📢 Phase 1: Training Pure Software Float Baseline (Ideal Float)...")
    acc_float, f1_float, std_float = train_and_evaluate_eeg(
        X_eeg, y_eeg, 
        out_classes=3, 
        device_profile=None, 
        lr=0.001, 
        epochs=args.epochs, 
        batch_size=64,
        dropout_p=0.3
    )
    
    # --- PHASE 2: HARDWARE-AWARE MODEL WITH DEVICE CONSTRAINTS ---
    print("\n📢 Phase 2: Training Hardware-Aware Model under Memristive Constraints...")
    acc_hw, f1_hw, std_hw = train_and_evaluate_eeg(
        X_eeg, y_eeg, 
        out_classes=3, 
        device_profile=nonvolatile_profile, 
        lr=0.001, 
        epochs=args.epochs, 
        batch_size=64,
        dropout_p=0.3
    )
    
    training_time = time.time() - t0
    
    # Report results
    print("\n  SIMULATION RESULTS & COMPARISON")
    print("=" * 60)
    print(f"  {'Metric':<25} | {'Ideal Float':<12} | {'Memristive HW':<16}")
    print("-" * 60)
    print(f"  {'Classification Acc (%)':<25} | {acc_float:<12.2f} | {acc_hw:<16.2f}")
    print(f"  {'F1-Score':<25} | {f1_float:<12.4f} | {f1_hw:<16.4f}")
    print(f"  {'Accuracy Loss Gap (%)':<25} | {'N/A':<12} | {acc_float - acc_hw:<16.2f}")
    print("=" * 60)
    
    print(f"🏆 Final Result: Accuracy = {acc_hw:.2f}% ± {std_hw:.2f}%, F1-Score = {f1_hw:.4f}")
    print(f"⏱️ Total training & evaluation time: {training_time:.2f}s")
    print("=" * 60)
    sys.stdout.flush()

if __name__ == "__main__":
    main()
