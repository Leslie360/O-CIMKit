import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiles.device_profile import DeviceProfile
from applications.face_rec.dataset import get_face_dataloaders
from applications.face_rec.model import FaceClassifier
from applications.face_rec.train import evaluate, calibrate_batchnorm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_model_drift(model, hours):
    """Recursively sets the drift_hours attribute of all custom bionic layers."""
    count = 0
    for module in model.modules():
        if hasattr(module, 'drift_hours'):
            module.drift_hours = float(hours)
            count += 1
    return count

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    profile_path = os.path.join(project_root, "profiles", "repository", "FingerMemristor.json")
    profile = DeviceProfile.from_json(profile_path) if os.path.exists(profile_path) else None
    
    print("=" * 60)
    print("CIM Platform - Device Reliability & Aging Robustness Study")
    print("=" * 60)
    if profile:
        print(f"  Loaded Device Profile: {profile.device_name}")
        print(f"  Conductance min/max: {profile.conductance_min} to {profile.conductance_max}")
    else:
        print("  ⚠️ FingerMemristor profile not found! Exiting.")
        return

    # Load dataloaders for Yale faces (real dataset)
    train_loader, test_loader, num_classes = get_face_dataloaders(dataset_name="yale", batch_size=16)
    
    # Phase 1: Train software float baseline (Ideal Float)
    print("📢 Phase 1: Training Software Float Baseline (Ideal Float)...")
    float_model = FaceClassifier(input_dim=4096, hidden_dim=256, num_classes=num_classes, device_profile=None).to(device)
    
    # Differential learning rates: 0.0002 for backbone, 0.002 for linear head
    backbone_params = []
    head_params = []
    for name, param in float_model.named_parameters():
        if "linear" in name or "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
            
    criterion = nn.CrossEntropyLoss()
    optimizer_float = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 0.0002, 'weight_decay': 1e-4},
        {'params': head_params, 'lr': 0.002, 'weight_decay': 1e-4}
    ])
    
    float_model.train()
    for epoch in range(25):
        loss_val = 0.0
        corr = 0
        tot = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer_float.zero_grad()
            out = float_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer_float.step()
            loss_val += loss.item()
            corr += out.argmax(dim=-1).eq(by).sum().item()
            tot += by.size(0)
        if (epoch + 1) % 5 == 0 or epoch == 24:
            print(f"    Epoch {epoch+1:02d} - Loss: {loss_val/len(train_loader):.4f}, Acc: {(corr/tot)*100.0:.2f}%")
            
    # Phase 2: Load weights into hardware-aware model and tune
    print("\n📢 Phase 2: Tuning Hardware-Aware Model under Crossbar constraints...")
    model = FaceClassifier(input_dim=4096, hidden_dim=256, num_classes=num_classes, device_profile=profile).to(device)
    model.load_state_dict(float_model.state_dict(), strict=False)
    
    # Differential learning rates for hardware tuning
    backbone_params_hw = []
    head_params_hw = []
    for name, param in model.named_parameters():
        if "linear" in name or "fc" in name:
            head_params_hw.append(param)
        else:
            backbone_params_hw.append(param)
            
    optimizer_hw = torch.optim.AdamW([
        {'params': backbone_params_hw, 'lr': 0.0001, 'weight_decay': 1e-4},
        {'params': head_params_hw, 'lr': 0.001, 'weight_decay': 1e-4}
    ])
    
    model.train()
    for epoch in range(15):
        loss_val = 0.0
        corr = 0
        tot = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer_hw.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer_hw.step()
            loss_val += loss.item()
            corr += out.argmax(dim=-1).eq(by).sum().item()
            tot += by.size(0)
        if (epoch + 1) % 5 == 0 or epoch == 14:
            print(f"    Epoch {epoch+1:02d} - Loss: {loss_val/len(train_loader):.4f}, Acc: {(corr/tot)*100.0:.2f}%")
            
    # Time points to evaluate (hours of operational lifetime)
    # 0 = Fresh, 1h, 12h, 24h (1 day), 168h (1 week), 720h (1 month), 8760h (1 year), 87600h (10 years)
    time_points = [0.0, 1.0, 12.0, 24.0, 168.0, 720.0, 8760.0, 87600.0]
    time_labels = ["Fresh", "1 Hour", "12 Hours", "1 Day", "1 Week", "1 Month", "1 Year", "10 Years"]
    
    accuracies = []
    
    print("\n⏳ Simulating crossbar weight drift and noise degradation over time...")
    for h, label in zip(time_points, time_labels):
        # Set drift hours on the model layers
        set_model_drift(model, h)
        
        # Calibrate Batch Normalization parameters for the current aging/drift state
        calibrate_batchnorm(model, train_loader, device)
        
        # Evaluate test accuracy using the standard evaluation function
        test_loss, acc = evaluate(model, test_loader, criterion)
        accuracies.append(acc)
        print(f"  [{label:<9}] Elapsed: {h:>7.1f} hours | Accuracy: {acc:.2f}%")
        
    print("=" * 60)
    print("📊 RELIABILITY REPORT TABLE")
    print("=" * 60)
    print("| Lifetime Duration | Sim Hours | Model Accuracy | Retention Status |")
    print("| :--- | :--- | :--- | :--- |")
    for label, h, acc in zip(time_labels, time_points, accuracies):
        status = "✅ Stable" if acc >= accuracies[0] - 5.0 else "⚠️ Degraded"
        print(f"| **{label}** | {h:.1f} | **{acc:.2f}%** | {status} |")
    print("=" * 60)
    
    # 1. Matplotlib Dark Theme Plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    fig.patch.set_facecolor('#0f0f12')
    ax.set_facecolor('#141419')
    
    x_indices = np.arange(len(time_points))
    
    # Plot line and points
    ax.plot(x_indices, accuracies, marker='o', color='#00f5d4', linewidth=2.5, markersize=8, label='Device Retention Curve')
    
    # Add fill under curve
    ax.fill_between(x_indices, accuracies, color='#00f5d4', alpha=0.1)
    
    # Annotate values
    for idx, acc in enumerate(accuracies):
        ax.annotate(f"{acc:.1f}%", 
                    xy=(idx, acc), 
                    xytext=(0, 10), 
                    textcoords="offset points", 
                    ha='center', va='bottom', 
                    fontsize=9, color='#e0e0e6', fontweight='bold')
        
    ax.set_title('⏳ Organic Memristive Crossbar Reliability & Aging Retention Study', fontsize=14, fontweight='bold', pad=20, color='#ffffff')
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold', labelpad=10, color='#e0e0e6')
    ax.set_xticks(x_indices)
    ax.set_xticklabels(time_labels, fontsize=10, fontweight='bold', color='#c0c0c6')
    ax.set_ylim(0, 115)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#33333d')
    ax.spines['bottom'].set_color('#33333d')
    ax.grid(axis='y', linestyle='--', alpha=0.15, color='#888899')
    
    plt.tight_layout()
    
    # Save files
    plot_path = os.path.join(project_root, "reports", "reliability_retention.png")
    plt.savefig(plot_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    print(f"✅ Reliability chart saved to: {plot_path}")
    
    # Copy to artifact folder
    artifact_dir = "/home/qiaosir/.gemini/antigravity-cli/brain/fec583e9-bdc3-4183-a617-20063af7c173"
    if os.path.exists(artifact_dir):
        import shutil
        shutil.copy(plot_path, os.path.join(artifact_dir, "reliability_retention.png"))
        print(f"✅ Reliability chart copied to artifact folder.")

if __name__ == "__main__":
    main()
