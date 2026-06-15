import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve

class VolatileReservoir:
    """Volatile reservoir (fast state, processing real-time signals)."""
    def __init__(self, tau=3.64, sigma=0.01):
        self.tau = tau
        self.sigma = sigma
        self.alpha = np.exp(-1.0 / tau)
        self.state = 0.0

    def reset(self):
        self.state = 0.0

    def step(self, x_t):
        u_t = self.alpha * self.state + (1.0 - self.alpha) * x_t
        u_t += self.sigma * np.random.randn()
        self.state = np.clip(u_t, -10.0, 10.0)
        return self.state

class NonVolatileMemory:
    """Non-volatile memory (slow state, storing personalized baseline)."""
    def __init__(self, n_states=64):
        self.n_states = n_states
        self.states = np.zeros(n_states)
        self.current_idx = 0

    def write(self, value):
        """Write state quantized to the nearest level."""
        quantized = np.round(value * (self.n_states - 1)) / (self.n_states - 1)
        self.states[self.current_idx] = quantized
        self.current_idx = (self.current_idx + 1) % self.n_states

    def read(self):
        return self.states.copy()

class EdgeLLMSentinel:
    """
    Edge-LLM Sentinel: Ultra-low power semantic interceptor.
    - Volatile (tau = 3.64s): filters temporal abnormalities.
    - Non-volatile (64 states): keeps baseline template memory.
    """
    def __init__(self, device_profile=None, tau=3.64, n_states=64):
        self.profile = device_profile
        
        # Load parameters from device profile if available
        if self.profile:
            tau = self.profile.tau_volatile if self.profile.tau_volatile is not None else tau
            n_states = self.profile.discrete_states_count if self.profile.discrete_states_count is not None else n_states

        self.reservoir = VolatileReservoir(tau)
        self.memory = NonVolatileMemory(n_states)
        self.classifier = None
        self.threshold = 0.5

    def extract_features(self, signal):
        self.reservoir.reset()
        states = []
        for x in signal:
            state = self.reservoir.step(x)
            states.append(state)
        states = np.array(states)

        # Multi-dimensional statistics
        features = [
            np.mean(states),
            np.std(states),
            np.max(states),
            np.min(states),
            np.percentile(states, 25),
            np.percentile(states, 75),
            np.mean(np.diff(states)),
            np.std(np.diff(states)),
            np.sum(np.abs(states) > np.std(states) * 2.0),
        ]
        return features

    def train(self, normal_signals, anomaly_signals):
        print("  Training Sentinel classifier...")
        X_normal = [self.extract_features(sig) for sig in normal_signals]
        X_anomaly = [self.extract_features(sig) for sig in anomaly_signals]

        X = np.array(X_normal + X_anomaly)
        y = np.array([0] * len(X_normal) + [1] * len(X_anomaly))

        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X, y)

        # Compute optimal threshold based on precision-recall curve
        y_prob = self.classifier.predict_proba(X)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y, y_prob)
        
        f1_scores = 2.0 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        # Handle edge case where thresholds is shorter than precision/recall
        if best_idx < len(thresholds):
            self.threshold = thresholds[best_idx]
        else:
            self.threshold = 0.5
            
        print(f"  Sentinel classifier trained. Optimal threshold: {self.threshold:.4f}")
        return self.classifier

    def detect(self, signal):
        if self.classifier is None:
            raise RuntimeError("❌ Classifier has not been trained yet.")
        features = self.extract_features(signal)
        features = np.array(features).reshape(1, -1)
        prob = self.classifier.predict_proba(features)[0, 1]
        is_anomaly = prob > self.threshold
        return is_anomaly, prob
