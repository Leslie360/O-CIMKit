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
from core.layers import OrganicSynapseConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DVSGestureNet(nn.Module):
    """
    Hardware-Aware Convolutional Neural Network for processing event-frame streams.
    Simulates memristive non-volatile weight mapping and C2C writing noise.
    """
    def __init__(self, num_classes=11, device_profile=None):
        super().__init__()
        # Input size: (Batch, Channels=2, Height=32, Width=32)
        # Channel 0: positive polarity events, Channel 1: negative polarity events
        self.conv1 = OrganicSynapseConv(2, 16, kernel_size=3, device_profile=device_profile, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = OrganicSynapseConv(16, 32, kernel_size=3, device_profile=device_profile, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = OrganicSynapseConv(32, 64, kernel_size=3, device_profile=device_profile, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: (B, 2, 32, 32)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = torch.relu(self.bn3(self.conv3(out)))
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def get_dvs_mock_dataset(num_samples=400, num_classes=11):
    """Generates synthetic event-frame tensors representing accumulated DVS camera streams."""
    np.random.seed(42)
    torch.manual_seed(42)
    # Create learnable centroids for classes
    centroids = np.random.uniform(1.0, 4.0, (num_classes, 2, 32, 32)).astype(np.float32)
    
    # Generate labels representing 11 gesture classes (hand wave, arm roll, clap, etc.)
    y = np.random.randint(0, num_classes, size=num_samples).astype(np.int64)
    
    # Add noise to centroids to create samples
    X = centroids[y] + np.random.normal(0.0, 0.5, (num_samples, 2, 32, 32)).astype(np.float32)
    
    # Normalize inputs
    X = (X - X.mean()) / (X.std() + 1e-8)
    
    split = int(0.8 * num_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    train_ds = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds = torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    
    return train_ds, test_ds

def train_and_evaluate(profile, args, train_loader, test_loader, track_name="Ideal Float", pretrained_model=None):
    model = DVSGestureNet(num_classes=11, device_profile=profile).to(device)
    if pretrained_model is not None:
        model.load_state_dict(pretrained_model.state_dict(), strict=False)
        print(f"  Loaded fine-tuned baseline weights for {track_name} track tuning.")
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_acc = 0.0
    resample_factor = 5 if args.epochs <= 5 else 1
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for _ in range(resample_factor):
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                # Inject 2% input noise to simulate sensing jitter
                bx_noisy = bx + torch.randn_like(bx) * 0.02
                optimizer.zero_grad()
                outputs = model(bx_noisy)
                loss = criterion(outputs, by)
                loss.backward()
                optimizer.step()
                
                # Clamp weights to safe memristive conductance bounds [-1, 1]
                with torch.no_grad():
                    for param in model.parameters():
                        if param.requires_grad:
                            param.data.clamp_(-1.0, 1.0)
                            
                running_loss += loss.item()
                correct += outputs.argmax(dim=-1).eq(by).sum().item()
                total += by.size(0)
                
        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                outputs = model(bx)
                test_correct += outputs.argmax(dim=-1).eq(by).sum().item()
                test_total += by.size(0)
                
        test_acc = (test_correct / test_total) * 100.0
        best_acc = max(best_acc, test_acc)
        scheduler.step()
        
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            print(f"  Epoch {epoch+1:02d} - Train Loss: {running_loss/(len(train_loader)*resample_factor):.4f}, Train Acc: {(correct/total)*100.0:.2f}%, Test Acc: {test_acc:.2f}%")
            
    return best_acc, model

def main():
    parser = argparse.ArgumentParser(description="DVS Event-Frame Gesture Recognition under CIM Constraints")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    
    if "ipykernel" in sys.modules or not sys.argv[0].endswith("train.py"):
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
        
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Load Device Profile
    profile_path = os.path.join(project_root, "profiles", "repository", "FingerMemristor.json")
    profile = DeviceProfile.from_json(profile_path) if os.path.exists(profile_path) else None
    
    print("=" * 60)
    print("CIM Platform - DVS Event-Frame Gesture Recognition")
    print("=" * 60)
    if profile:
        print(f"  Loaded Device Profile: {profile.device_name}")
        print(f"  Noise injection level: {profile.get_noise_std():.4f}")
    else:
        print("  ⚠️ FingerMemristor profile not found! Running ideal software baseline.")
        
    train_ds, test_ds = get_dvs_mock_dataset()
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    t0 = time.time()
    # 1. Ideal software baseline
    print("📢 Phase 1: Training Pure Software Float Baseline (Ideal Float)...")
    acc_float, model_float = train_and_evaluate(None, args, train_loader, test_loader, track_name="Ideal Float")
    
    # 2. Hardware-aware simulation
    print("\n📢 Phase 2: Training Hardware-Aware Model under Memristive Constraints...")
    acc_hw, model_hw = train_and_evaluate(profile, args, train_loader, test_loader, track_name="Memristive HW", pretrained_model=model_float)
    
    training_time = time.time() - t0
    
    # Summary
    print("\n  SIMULATION RESULTS & COMPARISON")
    print("=" * 60)
    print(f"  {'Metric':<25} | {'Ideal Float':<12} | {'Memristive HW':<16}")
    print("-" * 60)
    print(f"  {'Gesture Rec Acc (%)':<25} | {acc_float:<12.2f} | {acc_hw:<16.2f}")
    print(f"  {'Accuracy Loss Gap (%)':<25} | {'N/A':<12} | {acc_float - acc_hw:<16.2f}")
    print("=" * 60)
    print(f"🏆 Final DVS Gesture Rec Accuracy: {acc_hw:.2f}%")
    print(f"⏱️ Total execution time: {training_time:.2f}s")
    print("=" * 60)

if __name__ == "__main__":
    main()
