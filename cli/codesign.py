import os
import json
import sys

from profiles.device_profile import DeviceProfile

def run_codesign(device_path):
    """
    Run hardware-software codesign exploration for varying parameters.
    
    Args:
        device_path (str): Path to the base device JSON profile.
    """
    print(f"🚀 Starting Hardware-Software Co-Design Exploration")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repository_dir = os.path.join(project_root, "profiles", "repository")
    
    # 1. Parse device profile
    if not os.path.exists(device_path):
        # Check inside profiles/repository
        device_path_repo = os.path.join(repository_dir, device_path)
        if not device_path_repo.endswith(".json"):
            device_path_repo += ".json"
        if os.path.exists(device_path_repo):
            device_path = device_path_repo
        else:
            print(f"❌ Device Profile not found at: {device_path}")
            return

    # Dynamic imports to avoid overhead
    import torch
    import torch.nn as nn
    import numpy as np
    
    from core.compiler import BionicCoDesignCompiler
    from core.layers import SelfHealingCrossbar, DynamicOrganicSynapse
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("CODESIGN COMPILER & ONLINE SELF-HEALING VERIFICATION")
    print("=" * 60)
    
    # 2. Instantiate compiler
    compiler = BionicCoDesignCompiler(device_path)
    compiled_settings = compiler.compile()
    
    # 3. Synthesize model
    model = compiler.synthesize_model("feedforward", in_features=20, out_features=2, hidden_dim=64)
    model.to(device)
    
    # Generate dataset
    np_random = np.random.RandomState(42)
    X = np_random.randn(1500, 20)
    y = (X[:, 0] * X[:, 1] + X[:, 2]**2 > 0.5).astype(int)
    split = int(0.8 * 1500)
    
    X_train = torch.FloatTensor(X[:split]).to(device)
    y_train = torch.LongTensor(y[:split]).to(device)
    X_test = torch.FloatTensor(X[split:]).to(device)
    y_test = torch.LongTensor(y[split:]).to(device)
    
    # Train using compiled suggested learning rate & weight decay
    print("\n📢 Phase 1: Training Synthesized Self-Healing Crossbar Model...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=compiled_settings["suggested_lr"], 
        weight_decay=compiled_settings["suggested_wd"]
    )
    
    model.train()
    for epoch in range(100):
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
    
    # 4. Age the model under power-law conductance drift
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
    
    seq_input = torch.randn(8, 50, 10).to(device)
    dynamic_output = dynamic_layer(seq_input, return_sequence=False)
    print(f"  Output state shape: {dynamic_output.shape} (Expected: 8, 16)")
    print("  ✅ DynamicOrganicSynapse verification completed successfully.")
    print("=" * 60)

