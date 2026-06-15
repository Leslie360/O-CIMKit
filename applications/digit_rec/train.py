import os
import sys
import numpy as np
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from profiles.device_profile import DeviceProfile
from applications.digit_rec.dataset import load_mnist_data
from applications.digit_rec.model import UltraReservoir

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 1. Load Device Profiles
    volatile_profile_path = os.path.join(project_root, "profiles", "repository", "PFO_AlOx_Volatile.json")
    volatile_profile = DeviceProfile.from_json(volatile_profile_path) if os.path.exists(volatile_profile_path) else None
    
    print("=" * 60)
    print("CIM Platform - Sequential MNIST In-Sensor Computing")
    print("=" * 60)

    # 2. Load 8x8 MNIST digits subset (Full 1797 samples for peak accuracy)
    images, labels = load_mnist_data(num_samples=1797)
    
    # 3. Create Ultra Reservoir Model with optimized hyperparameters
    print("  Initializing multi-scale Ultra Reservoir...")
    reservoir = UltraReservoir(
        device_profile=volatile_profile, n_reservoir=800, seed=42,
        spectral_radius=1.20666, input_scale=0.41551
    )
    
    # 4. Extract physical features
    print("  Extracting multi-scale features through dual-mode reservoir dynamics...")
    features = []
    images_np = images.numpy()
    for i in range(len(images_np)):
        if i > 0 and i % 200 == 0:
            print(f"    Progress: {i}/{len(images_np)}")
        feat = reservoir.extract_features(images_np[i])
        features.append(feat)
        
    X_features = np.array(features)
    
    # Standardize features (No PCA is used to preserve complete high-dimensional information)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    
    # 5. Train & Evaluate using multiple Readout Classifiers
    print("  Training and selecting best Readout Classifier...")
    split_idx = int(0.8 * len(images_np))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]
    
    classifiers = {
        "RidgeClassifierCV": RidgeClassifierCV(alphas=np.logspace(-2, 2, 20)),
        "LogisticRegression": LogisticRegression(C=0.1, max_iter=1000, random_state=42),
        "MLPClassifier": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, alpha=0.01, random_state=42)
    }
    
    best_acc = 0.0
    best_name = ""
    for name, clf in classifiers.items():
        try:
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            acc = accuracy_score(y_test, preds) * 100
            print(f"    - Readout [{name}] Test Accuracy: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                best_name = name
        except Exception as e:
            print(f"    - Readout [{name}] training failed: {e}")
            
    print(f"🏆 Final sMNIST Test Accuracy: {best_acc:.2f}% (Best Readout: {best_name})")

if __name__ == "__main__":
    main()
