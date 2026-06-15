import os
import sys
import numpy as np
from sklearn.linear_model import Ridge

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from profiles.device_profile import DeviceProfile
from applications.chaotic_lorenz.dataset import generate_lorenz_1d
from applications.chaotic_lorenz.model import UltimateLorenzPredictor

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 1. Load Device Profiles
    volatile_profile_path = os.path.join(project_root, "profiles", "repository", "PFO_AlOx_Volatile.json")
    volatile_profile = DeviceProfile.from_json(volatile_profile_path) if os.path.exists(volatile_profile_path) else None
    
    print("=" * 60)
    print("CIM Platform - Lorenz Chaotic Attractor Prediction (Optimized)")
    print("=" * 60)

    # 2. Instantiate Predictor
    predictor = UltimateLorenzPredictor(device_profile=volatile_profile, n_nodes=200)

    # 3. Generate Lorenz System 1D Data
    print("  Generating Lorenz trajectory (4000 steps)...")
    lorenz_data = generate_lorenz_1d(steps=4000)
    
    train_len = 2000
    test_len = 1000
    delay = 5

    # 4. Extract & Augment Features
    print("  Extracting and augmenting reservoir states (delay=5)...")
    raw_states = predictor.simulate_reservoir(lorenz_data)
    aug_states = predictor.augment_states(raw_states, delay=delay)
    
    # Target is prediction of the next timestep (shift by 1)
    target_data = lorenz_data[delay + 1:]
    # Match lengths
    aug_states = aug_states[:-1]
    
    # Train / Test splits
    X_train = aug_states[:train_len]
    Y_train = target_data[:train_len]
    X_test = aug_states[train_len:train_len+test_len]
    Y_test = target_data[train_len:train_len+test_len]
    
    print(f"  Split dataset: Train={X_train.shape}, Test={X_test.shape}")

    # 5. Train Ridge Classifier
    print("  Training output layer using Ridge Regression (alpha=1e-4)...")
    clf = Ridge(alpha=1e-4)
    clf.fit(X_train, Y_train)
    
    # 6. Predict & Evaluate
    print("  Evaluating predictions NRMSE...")
    predictions = clf.predict(X_test)
    
    mse = np.mean((predictions - Y_test)**2)
    variance = np.var(Y_test)
    nrmse = np.sqrt(mse / variance) * 100
    
    print(f"🏆 Final Test NRMSE: {nrmse:.4f}%")
    if nrmse < 1.0:
        print("  ✅ Success: Prediction NRMSE successfully suppressed below 1%!")

if __name__ == "__main__":
    main()
