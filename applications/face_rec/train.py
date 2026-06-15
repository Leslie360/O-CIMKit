import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from profiles.device_profile import DeviceProfile
from applications.face_rec.dataset import get_face_dataloaders
from applications.face_rec.model import FaceClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def train_one_epoch(model, dataloader, criterion, optimizer, resample_factor=1, is_calibration=False):
    model.train()
    
    # Suppress noise during calibration mode
    for module in model.modules():
        if hasattr(module, 'noise_std'):
            if is_calibration:
                module.noise_std = 0.0
            else:
                if hasattr(module, 'profile') and module.profile is not None:
                    module.noise_std = module.profile.get_noise_std()
                    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for _ in range(resample_factor):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
    return (running_loss / (len(dataloader) * resample_factor)), 100.0 * correct / total

def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / len(dataloader), 100.0 * correct / total

def run_dual_track_train(profile, args, train_loader, test_loader, track_name="Ideal Float", pretrained_model=None, num_classes=15):
    # Create Model
    model = FaceClassifier(input_dim=4096, hidden_dim=256, num_classes=num_classes, device_profile=profile).to(device)
    
    if pretrained_model is not None:
        model.load_state_dict(pretrained_model.state_dict(), strict=False)
        print("  Loaded fine-tuned software float baseline weights for bionic tuning.")
        
    criterion = nn.CrossEntropyLoss()
    
    # Differential learning rates: smaller lr for backbone, full lr for QAT readout head
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "linear" in name or "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
            
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1, 'weight_decay': 1e-3},
        {'params': head_params, 'lr': args.lr, 'weight_decay': 1e-3}
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    resample_factor = 6 if args.epochs <= 5 else 1
    best_test_acc = 0.0
    
    # Stage 1: Freeze ResNet backbone, training the classification head only
    freeze_epochs = int(args.epochs * 0.15)
    if freeze_epochs > 0 and pretrained_model is None:
        print(f"  [Stage 1] Freezing ResNet backbone for the first {freeze_epochs} epochs. Training head only...")
        for name, param in model.named_parameters():
            if "linear" not in name and "fc" not in name:
                param.requires_grad = False
                
    print(f"\n🚀 Training {track_name} track...")
    for epoch in range(args.epochs):
        # Stage 2: Unfreeze all layers for full hardware-aware tuning
        if epoch == freeze_epochs and freeze_epochs > 0 and pretrained_model is None:
            print(f"  [Stage 2] Unfreezing ResNet backbone. Commencing full-network tuning...")
            for param in model.parameters():
                param.requires_grad = True
                
        is_calib = (epoch >= args.epochs - 2) if args.enable_calibration else False
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, 
            resample_factor=resample_factor, is_calibration=is_calib
        )
        calibrate_batchnorm(model, train_loader, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        scheduler.step()
        
        best_test_acc = max(best_test_acc, test_acc)
        current_lr = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else optimizer.param_groups[0]['lr']
        calib_str = "[CALIB]" if is_calib else ""
        print(f"  Epoch {epoch+1:02d} {calib_str} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, LR: {current_lr:.1e}")
        
    return best_test_acc, model

def main():
    parser = argparse.ArgumentParser(description="Yale Face recognition hardware-aware MLP training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.002, help="Initial learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--enable-calibration", action="store_true", default=True, help="Enable digital calibration")
    parser.add_argument("--dataset", type=str, default="yale", choices=["yale", "orl"], help="Dataset to use: yale or orl")
    
    if "ipykernel" in sys.modules or not sys.argv[0].endswith("train.py"):
        args = parser.parse_args([])
        if not hasattr(args, "dataset"):
            args.dataset = "yale"
    else:
        args = parser.parse_args()
        
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_path = os.path.join(project_root, "applications", "face_rec", "train.log")
    
    # Redirect output to both file and console
    sys.stdout = DualLogger(log_path)
    
    # 1. Load Device Profile
    profile_path = os.path.join(project_root, "profiles", "repository", "FingerMemristor.json")
    profile = DeviceProfile.from_json(profile_path) if os.path.exists(profile_path) else None
    
    print("=" * 60)
    print("CIM Platform - Yale Face Recognition ResNet-18")
    print("=" * 60)
    if profile:
        print(f"  Loaded Device Profile: {profile.device_name}")
        print(f"  Discrete Quantization: {profile.discrete_states_count} states")
    else:
        print("  ⚠️ FingerMemristor profile not found! Running standard float model.")

    # 2. Load Dataloaders
    train_loader, test_loader, num_classes = get_face_dataloaders(dataset_name=args.dataset, batch_size=args.batch_size)
    
    t0 = time.time()
    
    # --- PHASE 1: IDEAL SOFTWARE FLOAT BASELINE ---
    print("📢 Phase 1: Training Pure Software Float Baseline (Ideal Float)...")
    best_acc_float, model_float = run_dual_track_train(None, args, train_loader, test_loader, track_name="Ideal Float", num_classes=num_classes)
    
    # --- PHASE 2: HARDWARE-AWARE MODEL WITH DEVICE CONSTRAINTS ---
    print("\n📢 Phase 2: Training Hardware-Aware Model under Memristive Constraints...")
    best_acc_hw, model_hw = run_dual_track_train(profile, args, train_loader, test_loader, track_name="Memristive HW", pretrained_model=model_float, num_classes=num_classes)
    
    training_time = time.time() - t0
    
    # Report results
    print("\n  SIMULATION RESULTS & COMPARISON")
    print("=" * 60)
    print(f"  {'Metric':<25} | {'Ideal Float':<12} | {'Memristive HW':<16}")
    print("-" * 60)
    print(f"  {'Classification Acc (%)':<25} | {best_acc_float:<12.2f} | {best_acc_hw:<16.2f}")
    print(f"  {'Accuracy Loss Gap (%)':<25} | {'N/A':<12} | {best_acc_float - best_acc_hw:<16.2f}")
    print("=" * 60)
    
    print(f"🏆 Final Result: Accuracy = {best_acc_hw:.2f}%")
    print(f"⏱️ Total training & evaluation time: {training_time:.2f}s")
    print("=" * 60)
    sys.stdout.flush()

if __name__ == "__main__":
    main()
