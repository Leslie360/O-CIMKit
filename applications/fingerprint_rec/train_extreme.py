import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from profiles.device_profile import DeviceProfile
from applications.fingerprint_rec.dataset import get_dataloaders
from applications.fingerprint_rec.model import get_organic_resnet18

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


def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)

def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch, args, is_calibration=False):
    model.train()
    
    # In calibration mode, temporarily suppress noise
    if is_calibration and args.enable_calibration:
        for module in model.modules():
            if hasattr(module, 'noise_std'):
                module.noise_std = 0.0
                
    # Dynamic noise injection if enabled
    if args.enable_dynamic_noise and not is_calibration:
        base_noise = model.fc.weight.new_tensor(0.0) # dummy
        for module in model.modules():
            if hasattr(module, 'profile') and module.profile is not None:
                base_noise = module.profile.get_noise_std()
                break
        if base_noise == 0.0:
            base_noise = 2e-11 # fallback
            
        if epoch < args.epochs * 0.8:
            noise_mult = np.random.uniform(0.5, 2.5)
        else:
            noise_mult = np.random.uniform(0.5, 1.0)
            
        current_noise = base_noise * noise_mult
        for module in model.modules():
            if hasattr(module, 'noise_std'):
                module.noise_std = current_noise

    running_loss = 0.0
    correct = 0
    total = 0
    use_mixup = args.enable_mixup and (epoch >= args.warmup_epochs) and (not is_calibration)
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)
            
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            outputs = model(inputs)
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        
        # Clamp weights to safe range [-1, 1]
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.data.clamp_(-1.0, 1.0)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), 100. * correct / total

def main():
    parser = argparse.ArgumentParser(description="Extreme Organic Memristor Fingerprint ResNet-18 Training")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--warmup-epochs", type=int, default=10, help="Warmup epochs before mixup")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--enable-mixup", action="store_true", default=True, help="Enable mixup augmentation")
    parser.add_argument("--enable-dynamic-noise", action="store_true", default=True, help="Enable dynamic noise injection")
    parser.add_argument("--enable-calibration", action="store_true", default=True, help="Enable digital calibration")
    
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_path = os.path.join(project_root, "applications", "fingerprint_rec", "extreme_train.log")
    
    # Redirect print to log file and stdout
    sys.stdout = DualLogger(log_path)
    
    # 1. Load Device Profile
    profile_path = os.path.join(project_root, "profiles", "repository", "FingerMemristor.json")
    profile = DeviceProfile.from_json(profile_path) if os.path.exists(profile_path) else None
    
    print("=" * 60)
    print("CIM Platform - Fingerprint Recognition ResNet-18 (EXTREME TUNING)")
    print("=" * 60)
    if profile:
        print(f"  Loaded Device Profile: {profile.device_name}")
        print(f"  Conductance range: {profile.conductance_min} to {profile.conductance_max}")
    else:
        print("  ⚠️ FingerMemristor profile not found! Running standard float model.")
        
    # 2. Get dataloaders
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)
    
    # 3. Create Model
    model = get_organic_resnet18(device_profile=profile, num_classes=5, pretrained=True).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print(f"  Training for {args.epochs} epochs...")
    
    # Stage 1: Freeze ResNet backbone, training the classification head only
    freeze_epochs = int(args.epochs * 0.10)  # e.g., 15 epochs for 150 epochs
    if freeze_epochs > 0:
        print(f"  [Stage 1] Freezing ResNet backbone for the first {freeze_epochs} epochs. Training head only...")
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
                
    best_acc = 0.0
    for epoch in range(args.epochs):
        # Stage 2: Unfreeze all layers for full hardware-aware tuning
        if epoch == freeze_epochs and freeze_epochs > 0:
            print(f"  [Stage 2] Unfreezing ResNet backbone. Commencing full-network hardware-aware tuning...")
            for param in model.parameters():
                param.requires_grad = True
                
        is_calib = (epoch >= args.epochs - 2) if args.enable_calibration else False
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, epoch, args, is_calibration=is_calib)
        
        # Dynamically calibrate Batch Normalization to absorb device offsets prior to validation
        if args.enable_calibration:
            calibrate_batchnorm(model, train_loader, device)
            
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        scheduler.step()
        
        calib_str = "[CALIB]" if is_calib else ""
        print(f"  Epoch {epoch+1:02d} {calib_str} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            # Save the best model
            torch.save(model.state_dict(), os.path.join(project_root, "applications", "fingerprint_rec", "best_extreme_model.pth"))
            
    print("=" * 60)
    print(f"🏆 Extreme training finished! Best Test Accuracy: {best_acc:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
