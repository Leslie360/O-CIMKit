import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from profiles.device_profile import DeviceProfile
from applications.cifar10_vision.dataset import get_dataloaders
from applications.cifar10_vision.model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESPONSIVITY_CONST = None
MEAN_CONST = None
STD_CONST = None

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

def apply_gpu_physical_optic_transform(inputs, device, max_photons=10000.0, is_float_baseline=False):
    """GPU-parallelized physical optic capture and normalization for speed."""
    global RESPONSIVITY_CONST, MEAN_CONST, STD_CONST
    if RESPONSIVITY_CONST is None:
        RESPONSIVITY_CONST = torch.tensor([0.706, 1.0, 0.758], device=device).view(1, 3, 1, 1)
        MEAN_CONST = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
        STD_CONST = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1, 3, 1, 1)

    if is_float_baseline:
        # Standard software normalization directly
        return (inputs - MEAN_CONST) / STD_CONST

    # 1. Inverse Gamma correction (gamma = 2.2)
    inputs_linear = torch.pow(inputs, 2.2)
    
    # 2. Spectral Responsivity
    inputs_effective = inputs_linear * RESPONSIVITY_CONST
    
    # 3. Poisson shot noise injection
    expected_electrons = inputs_effective * max_photons
    noisy_electrons = torch.poisson(torch.clamp(expected_electrons, min=0.0))
    
    # 4. Normalize back and clip
    physical_signal = noisy_electrons / max_photons
    physical_signal = torch.clamp(physical_signal, 0.0, 1.0)
    
    # 5. Apply torchvision standard normalization
    return (physical_signal - MEAN_CONST) / STD_CONST

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)

def calibrate_batchnorm(model, dataloader, device, max_photons=10000.0, is_float_baseline=False):
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
        for i, (inputs, _) in enumerate(dataloader):
            if i >= 100:  # limit to 100 batches for speed
                break
            inputs = inputs.to(device)
            inputs = apply_gpu_physical_optic_transform(inputs, device, max_photons=max_photons, is_float_baseline=is_float_baseline)
            _ = model(inputs)
            
    for name, module in model.named_modules():
        if name in old_noise_stds:
            module.noise_std = old_noise_stds[name]
        if name in old_momentums:
            module.momentum = old_momentums[name]

def train_one_epoch(net, dataloader, criterion, optimizer, scaler, epoch, total_epochs, resample_factor=1, is_calibration=False, max_photons=10000.0, use_mixup=True, is_float_baseline=False):
    net.train()
    
    # Suppress noise during calibration mode
    for module in net.modules():
        if hasattr(module, 'noise_std'):
            if is_calibration:
                module.noise_std = 0.0
            else:
                if hasattr(module, 'profile') and module.profile is not None:
                    module.noise_std = module.profile.get_noise_std()
                    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Mixup active between 10% and 90% of epochs, and not in calibration mode
    mixup_active = use_mixup and (epoch >= int(total_epochs * 0.1)) and (epoch < int(total_epochs * 0.9)) and (not is_calibration)
    
    for _ in range(resample_factor):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = apply_gpu_physical_optic_transform(inputs, device, max_photons=max_photons, is_float_baseline=is_float_baseline)
            optimizer.zero_grad()
            
            if mixup_active:
                mixed_inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=1.0)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = net(mixed_inputs)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) 
            scaler.step(optimizer)
            scaler.update()
            
            # Clamp weights to safe range [-1, 1]
            with torch.no_grad():
                for param in net.parameters():
                    if param.requires_grad:
                        param.data.clamp_(-1.0, 1.0)
                        
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
    return (running_loss / (len(dataloader) * resample_factor)), 100.0 * correct / total

def evaluate(net, dataloader, criterion, max_photons=10000.0, is_float_baseline=False):
    net.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = apply_gpu_physical_optic_transform(inputs, device, max_photons=max_photons, is_float_baseline=is_float_baseline)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = net(inputs)
                loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / len(dataloader), 100.0 * correct / total

def run_dual_track_train(profile, args, train_loader, val_loader, track_name="Ideal Float", is_float_baseline=False, pretrained_model=None):
    num_classes = 100 if getattr(args, "dataset", "cifar10") == "cifar100" else 10
    net = get_model(device_profile=profile, num_classes=num_classes).to(device)
    
    if pretrained_model is not None:
        net.load_state_dict(pretrained_model.state_dict(), strict=False)
        print("  Loaded fine-tuned software float baseline weights for bionic tuning.")
        
    criterion = nn.CrossEntropyLoss()
    
    # Differential learning rates: smaller lr for pretrained backbone, full lr for head
    backbone_params = []
    head_params = []
    for name, param in net.named_parameters():
        if "linear" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
            
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1, 'weight_decay': 1e-4},
        {'params': head_params, 'lr': args.lr, 'weight_decay': 1e-4}
    ])
    scaler = torch.amp.GradScaler('cuda')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    resample_factor = 6 if args.epochs <= 5 else 1
    best_val_acc = 0.0
    
    # Stage 1: Freeze ResNet backbone, training the classification head only
    freeze_epochs = int(args.epochs * 0.15)  # e.g., 15 epochs for 100 epochs
    if freeze_epochs > 0 and pretrained_model is None:
        print(f"  [Stage 1] Freezing ResNet backbone for the first {freeze_epochs} epochs. Training head only...")
        for name, param in net.named_parameters():
            if "linear" not in name:
                param.requires_grad = False
                
    print(f"\n🚀 Training {track_name} track...")
    for epoch in range(args.epochs):
        # Stage 2: Unfreeze all layers for full hardware-aware tuning
        if epoch == freeze_epochs and freeze_epochs > 0 and pretrained_model is None:
            print(f"  [Stage 2] Unfreezing ResNet backbone. Commencing full-network tuning...")
            for param in net.parameters():
                param.requires_grad = True
                
        is_calib = (epoch >= args.epochs - 2) if args.enable_calibration else False
        
        train_loss, train_acc = train_one_epoch(
            net, train_loader, criterion, optimizer, scaler, 
            epoch=epoch, total_epochs=args.epochs, resample_factor=resample_factor,
            is_calibration=is_calib, max_photons=args.max_photons, 
            use_mixup=True, is_float_baseline=is_float_baseline
        )
        if is_calib or (epoch == args.epochs - 1):
            calibrate_batchnorm(net, train_loader, device, max_photons=args.max_photons, is_float_baseline=is_float_baseline)
            val_loss, val_acc = evaluate(net, val_loader, criterion, max_photons=args.max_photons, is_float_baseline=is_float_baseline)
        elif epoch % 5 == 0:
            val_loss, val_acc = evaluate(net, val_loader, criterion, max_photons=args.max_photons, is_float_baseline=is_float_baseline)
        else:
            val_loss, val_acc = -1.0, -1.0
        scheduler.step()
        
        if val_acc > 0.0:
            best_val_acc = max(best_val_acc, val_acc)
            
        calib_str = "[CALIB]" if is_calib else ""
        print(f"  Epoch {epoch+1:02d} {calib_str} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
    return best_val_acc, net

def main():
    parser = argparse.ArgumentParser(description="OECT device spectral vision CNN CIFAR-10 training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--max-photons", type=float, default=10000.0, help="Max photons (10000=bright, 20=dark)")
    parser.add_argument("--enable-calibration", action="store_true", default=True, help="Enable digital calibration")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"], help="Dataset to use: cifar10 or cifar100")
    
    if "ipykernel" in sys.modules or not sys.argv[0].endswith("train.py"):
        args = parser.parse_args([])
        # Set dataset default if empty
        if not hasattr(args, "dataset"):
            args.dataset = "cifar10"
    else:
        args = parser.parse_args()
        
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_path = os.path.join(project_root, "applications", "cifar10_vision", "extreme_train.log")
    
    # Redirect print to log file and stdout
    sys.stdout = DualLogger(log_path)
    
    # 1. Load Device Profile
    profile_path = os.path.join(project_root, "profiles", "repository", "OECT_Vision.json")
    profile = DeviceProfile.from_json(profile_path) if os.path.exists(profile_path) else None
    
    print("=" * 60)
    print("CIM Platform - CIFAR-10 Bionic Organic Vision System")
    print("=" * 60)
    if profile:
        print(f"  Loaded Device Profile: {profile.device_name}")
        print(f"  OECT currents: {profile.conductance_min} to {profile.conductance_max}")
    else:
        print("  ⚠️ OECT_Vision profile not found! Running standard float model.")
        
    # 2. Get dataloaders
    train_loader, val_loader = get_dataloaders(batch_size=args.batch_size, dataset_name=args.dataset)
    
    t0 = time.time()
    
    # --- PHASE 1: IDEAL SOFTWARE FLOAT BASELINE ---
    print("📢 Phase 1: Training Pure Software Float Baseline (Ideal Float)...")
    best_acc_float, net_float = run_dual_track_train(None, args, train_loader, val_loader, track_name="Ideal Float", is_float_baseline=True)
    
    # --- PHASE 2: HARDWARE-AWARE MODEL WITH DEVICE CONSTRAINTS ---
    print("\n📢 Phase 2: Training Hardware-Aware Model under Memristive Constraints...")
    best_acc_hw, net_hw = run_dual_track_train(profile, args, train_loader, val_loader, track_name="Memristive HW", is_float_baseline=False, pretrained_model=net_float)
    
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

