import os
import sys
import numpy as np
from sklearn.linear_model import Ridge

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.autotune import AutoTuner
from profiles.device_profile import DeviceProfile
from applications.chaotic_lorenz.dataset import generate_lorenz_1d

class TunableLorenzPredictor:
    """Lorenz Attractor Predictor with tunable hyperparameters for AutoTuner demonstration."""
    def __init__(self, n_nodes=100, seed=42):
        self.n_nodes = n_nodes
        self.seed = seed
        
    def evaluate(self, spectral_radius, input_scale, leaking_rate, ridge_alpha, data, train_len=1500, test_len=800, delay=5):
        # 1. Initialize reservoir weights based on tuning parameters
        np.random.seed(self.seed)
        W_in = (np.random.rand(self.n_nodes) * 2 - 1) * input_scale
        W_res = (np.random.rand(self.n_nodes, self.n_nodes) * 2 - 1) * 0.1
        
        # Scale spectral radius
        try:
            eigvals = np.linalg.eigvals(W_res)
            W_res = W_res / np.max(np.abs(eigvals)) * spectral_radius
        except Exception:
            pass
            
        # 2. Simulate reservoir dynamics
        states = []
        x_state = np.zeros(self.n_nodes)
        for u in data:
            pre_activation = W_in * u + np.dot(W_res, x_state)
            x_state = (1.0 - leaking_rate) * x_state + leaking_rate * np.tanh(pre_activation)
            states.append(x_state.copy())
        states = np.array(states)
        
        # 3. Augment states
        N = states.shape[0]
        aug_states = np.zeros((N - delay, states.shape[1] * 3))
        for i in range(delay, N):
            current = states[i]
            past = states[i - delay]
            aug_states[i - delay] = np.concatenate([current, past, current**2])
            
        # Target shift
        target_data = data[delay + 1:]
        aug_states = aug_states[:-1]
        
        # Train / Test split
        X_train = aug_states[:train_len]
        Y_train = target_data[:train_len]
        X_test = aug_states[train_len:train_len+test_len]
        Y_test = target_data[train_len:train_len+test_len]
        
        # 4. Fit Ridge Regression
        clf = Ridge(alpha=ridge_alpha)
        clf.fit(X_train, Y_train)
        
        # 5. Predict & Calculate NRMSE
        predictions = clf.predict(X_test)
        mse = np.mean((predictions - Y_test)**2)
        variance = np.var(Y_test)
        nrmse = np.sqrt(mse / variance) * 100
        return nrmse

def main():
    print("=" * 60)
    print("CIM Platform - AutoTuner Hyperparameter Search Demo")
    print("=" * 60)
    
    # 1. Load volatile device profile
    project_root = os.path.dirname(os.path.abspath(__file__))
    volatile_profile_path = os.path.join(project_root, "profiles", "repository", "PFO_AlOx_Volatile.json")
    if os.path.exists(volatile_profile_path):
        profile = DeviceProfile.from_json(volatile_profile_path)
        print(f"📖 Loaded Device Profile: {profile.device_name} (Volatile leaking rate default: {profile.leaking_rate_volatile})")
    else:
        print("⚠️ Device profile not found, proceeding with default parameters.")
        
    # 2. Generate Lorenz System Data
    print("📈 Generating Lorenz Attractor 1D data...")
    lorenz_data = generate_lorenz_1d(steps=3000)
    
    predictor = TunableLorenzPredictor(n_nodes=100)
    
    # 3. Define target function to maximize
    # Since AutoTuner seeks to MAXIMIZE accuracy, we maximize (100 - NRMSE)
    def target_accuracy_fn(spectral_radius, input_scale, leaking_rate, ridge_alpha):
        nrmse = predictor.evaluate(spectral_radius, input_scale, leaking_rate, ridge_alpha, lorenz_data)
        return 100.0 - nrmse
        
    # 4. Launch AutoTuner
    tuner = AutoTuner(target_accuracy_fn=target_accuracy_fn, n_trials=15)
    best_params, best_metric = tuner.tune()
    
    best_nrmse = 100.0 - best_metric
    print("=" * 60)
    print("🎯 Optimization Results Summary")
    print("-" * 60)
    print(f"  Best chaotic forecast NRMSE: {best_nrmse:.4f}%")
    print(f"  Best spectral_radius:        {best_params['spectral_radius']:.4f}")
    print(f"  Best input_scale:           {best_params['input_scale']:.4f}")
    print(f"  Best leaking_rate:          {best_params['leaking_rate']:.4f}")
    print(f"  Best ridge_alpha:           {best_params['ridge_alpha']:.4e}")
    print("=" * 60)

if __name__ == "__main__":
    main()
