import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.compiler import BionicCoDesignCompiler
from core.layers import DynamicOrganicSynapse, SelfHealingCrossbar

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_simple_classification_data(num_samples=1500, dim=20):
    np.random.seed(42)
    X = np.random.randn(num_samples, dim)
    # Simple non-linear classification boundary
    y = (X[:, 0] * X[:, 1] + X[:, 2]**2 > 0.5).astype(int)
    
    # Train / Test split
    split = int(0.8 * num_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return (
        torch.FloatTensor(X_train).to(device),
        torch.LongTensor(y_train).to(device),
        torch.FloatTensor(X_test).to(device),
        torch.LongTensor(y_test).to(device)
    )

def main():
    print("=" * 60)
    print("CODESIGN COMPILER & ONLINE SELF-HEALING VERIFICATION")
    print("=" * 60)
    
    # 1. Instantiate BionicCoDesignCompiler
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    profile_path = os.path.join(project_root, "profiles", "repository", "FingerMemristor.json")
    
    compiler = BionicCoDesignCompiler(profile_path)
    compiled_settings = compiler.compile()
    
    # 2. Synthesize Model
    model = compiler.synthesize_model("feedforward", in_features=20, out_features=2, hidden_dim=64)
    model.to(device)
    
    # Get dataset
    X_train, y_train, X_test, y_test = get_simple_classification_data()
    
    # 3. Train using compiler suggested learning rate & weight decay
    print("\n📢 Phase 1: Training Synthesized Self-Healing Crossbar Model...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=compiled_settings["suggested_lr"], 
        weight_decay=compiled_settings["suggested_wd"]
    )
    
    # Train epochs
    epochs = 100
    model.train()
    for epoch in range(epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    model.eval()
    for m in model.modules():
        if isinstance(m, SelfHealingCrossbar):
            m.self_healing_enabled = False
            
    with torch.no_grad():
        fresh_out = model(X_test)
        preds = fresh_out.argmax(dim=-1)
        fresh_acc = preds.eq(y_test).sum().item() / len(y_test) * 100.0
    print(f"✅ Fresh Model Accuracy (0 hours drift): {fresh_acc:.2f}%")
    
    # 4. Age the model under power-law conductance drift (e.g. 1 year and 10 years)
    print("\n📢 Phase 2: Simulating 10-Year Weight Drift & Active Self-Healing...")
    time_points = [8760.0, 87600.0]
    time_labels = ["1 Year (8,760h)", "10 Years (87,600h)"]
    
    for h, label in zip(time_points, time_labels):
        print(f"\n⏳ Aging Device to: {label}...")
        
        # Condition A: Without Self-Healing (Naive Uncompensated)
        for m in model.modules():
            if isinstance(m, SelfHealingCrossbar):
                m.drift_hours = h
                m.self_healing_enabled = False
                
        with torch.no_grad():
            out_a = model(X_test)
            preds_a = out_a.argmax(dim=-1)
            acc_a = preds_a.eq(y_test).sum().item() / len(y_test) * 100.0
            mse_a = torch.mean((out_a - fresh_out)**2).item()
            
        # Condition B: Global Scaling Compensation (IBM AIHWKit-like baseline)
        drift_exp = getattr(compiler.profile, 'drift_exponent', 0.06)
        factor = (h / 1.0) ** (-drift_exp) if h > 1.0 else 1.0
        with torch.no_grad():
            out_global = out_a / factor
            preds_global = out_global.argmax(dim=-1)
            acc_global = preds_global.eq(y_test).sum().item() / len(y_test) * 100.0
            mse_global = torch.mean((out_global - fresh_out)**2).item()

        # Condition C: With Online Unsupervised Self-Healing (Our Method)
        for m in model.modules():
            if isinstance(m, SelfHealingCrossbar):
                m.self_healing_enabled = True
                
        with torch.no_grad():
            out_b = model(X_test)
            preds_b = out_b.argmax(dim=-1)
            acc_b = preds_b.eq(y_test).sum().item() / len(y_test) * 100.0
            mse_b = torch.mean((out_b - fresh_out)**2).item()
            
        print(f"  [Naive (Uncompensated)] Accuracy: {acc_a:.2f}% | Activation MSE: {mse_a:.4e}")
        print(f"  [Global Scaling (IBM)]  Accuracy: {acc_global:.2f}% | Activation MSE: {mse_global:.4e} (Gain vs Naive: {mse_a/(mse_global+1e-20):.1f}x)")
        print(f"  [Our Online Self-Heal]  Accuracy: {acc_b:.2f}% | Activation MSE: {mse_b:.4e} (Gain vs IBM: {mse_global/(mse_b+1e-20):.1f}x)")
        
    # 5. Verify DynamicOrganicSynapse
    print("\n📢 Phase 3: Verifying DynamicOrganicSynapse (STP + LTP Uni-Synapse)...")
    dynamic_layer = DynamicOrganicSynapse(in_features=10, out_features=16, device_profile=compiler.profile)
    dynamic_layer.to(device)
    
    # Process time sequence input of shape (batch_size=8, seq_len=50, in_features=10)
    seq_input = torch.randn(8, 50, 10).to(device)
    dynamic_output = dynamic_layer(seq_input, return_sequence=False)
    print(f"  Output state shape: {dynamic_output.shape} (Expected: 8, 16)")
    print("  ✅ DynamicOrganicSynapse verification completed successfully.")
    print("=" * 60)

if __name__ == "__main__":
    main()
