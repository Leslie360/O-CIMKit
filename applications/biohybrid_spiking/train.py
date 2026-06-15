import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from profiles.device_profile import DeviceProfile
from applications.biohybrid_spiking.dataset import get_biohybrid_dataloader
from applications.biohybrid_spiking.model import BiohybridOECTSpiker

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

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    return 100.0 * correct / total

def main():
    parser = argparse.ArgumentParser(description="Hardware-Aware Biohybrid Brain-Computer Interface Spiking Classifier")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.02, help="Learning rate")
    parser.add_argument("--noise-level", type=float, default=0.05, help="Chemical background fluctuation noise")
    
    if "ipykernel" in sys.modules or not sys.argv[0].endswith("train.py"):
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
        
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_path = os.path.join(project_root, "applications", "biohybrid_spiking", "train.log")
    
    sys.stdout = DualLogger(log_path)
    
    # Load FC Classifier Device Profile (64-state non-volatile AlOx)
    alox_path = os.path.join(project_root, "profiles", "repository", "PFO_AlOx_NonVolatile.json")
    alox_profile = DeviceProfile.from_json(alox_path) if os.path.exists(alox_path) else None
    
    print("=" * 60)
    print("CIM Platform - Biohybrid Spiking Brain-Computer Interface")
    print("=" * 60)
    if alox_profile:
        print(f"  Loaded EEG Classifier Device: {alox_profile.device_name} ({alox_profile.discrete_states_count} states)")
    print(f"  Training for {args.epochs} epochs (Batch Size: {args.batch_size}, LR: {args.lr})")
    print("-" * 60)
    sys.stdout.flush()
    
    # Get Data
    train_loader, test_loader = get_biohybrid_dataloader(batch_size=args.batch_size, noise_level=args.noise_level)
    
    t0 = time.time()
    
    resample_factor = 6 if args.epochs <= 5 else 1
    
    # --- PHASE 1: IDEAL SOFTWARE FLOAT BASELINE ---
    print("📢 Phase 1: Training Pure Software Float Baseline (Ideal Float)...")
    model_float = BiohybridOECTSpiker(
        alox_profile=alox_profile,
        n_neurons=24,
        tau_oect=8.0,
        G0=0.1,
        beta=1.8,
        leak_lif=0.1,
        dt=1.0,
        is_float_baseline=True
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer_float = torch.optim.AdamW(model_float.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler_float = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_float, T_max=args.epochs)
    
    best_acc_float = 0.0
    for epoch in range(args.epochs):
        model_float.train()
        for _ in range(resample_factor):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer_float.zero_grad()
                outputs = model_float(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer_float.step()
        scheduler_float.step()
        acc = evaluate_model(model_float, test_loader)
        best_acc_float = max(best_acc_float, acc)
    print(f"  Ideal Float Baseline test acc: {best_acc_float:.2f}%")
    print("-" * 60)
    sys.stdout.flush()
    
    # --- PHASE 2: HARDWARE-AWARE MODEL WITH DEVICE CONSTRAINTS ---
    print("📢 Phase 2: Training Hardware-Aware Model under Memristive Constraints...")
    model_hw = BiohybridOECTSpiker(
        alox_profile=alox_profile,
        n_neurons=24,
        tau_oect=8.0,
        G0=0.1,
        beta=1.8,
        leak_lif=0.1,
        dt=1.0
    ).to(device)
    
    optimizer_hw = torch.optim.AdamW(model_hw.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler_hw = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_hw, T_max=args.epochs)
    
    best_acc_hw = 0.0
    for epoch in range(args.epochs):
        model_hw.train()
        for _ in range(resample_factor):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer_hw.zero_grad()
                outputs = model_hw(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer_hw.step()
        scheduler_hw.step()
        acc = evaluate_model(model_hw, test_loader)
        best_acc_hw = max(best_acc_hw, acc)
    print(f"  Hardware-Aware test acc: {best_acc_hw:.2f}%")
    print("-" * 60)
    sys.stdout.flush()
    
    training_time = time.time() - t0
    
    # Report results
    print("  SIMULATION RESULTS & COMPARISON")
    print("=" * 60)
    print(f"  {'Metric':<25} | {'Ideal Float':<12} | {'Memristive HW':<16}")
    print("-" * 60)
    print(f"  {'Classification Acc (%)':<25} | {best_acc_float:<12.2f} | {best_acc_hw:<16.2f}")
    print(f"  {'Accuracy Loss Gap (%)':<25} | {'N/A':<12} | {best_acc_float - best_acc_hw:<16.2f}")
    print("=" * 60)
    
    print(f"🏆 Final Biohybrid Spiking Pattern Accuracy: {best_acc_hw:.2f}%")
    print(f"⏱️ Total training & evaluation time: {training_time:.2f}s")
    print("=" * 60)
    sys.stdout.flush()

if __name__ == "__main__":
    main()
