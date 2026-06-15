import numpy as np

class VolatileReservoir:
    def __init__(self, tau=3.64, sigma=0.01):
        self.tau = tau
        self.sigma = sigma
        self.alpha = np.exp(-1.0 / tau)
        self.state = 0.0

    def reset(self):
        self.state = 0.0

    def get_response(self, input_val):
        self.state = self.alpha * self.state + (1.0 - self.alpha) * input_val
        self.state += self.sigma * np.random.randn()
        return self.state

class NonVolatileMemory:
    def __init__(self, n_states=64):
        self.n_states = n_states
        
    def encode(self, value):
        # Quantize to nearest discrete level
        quantized = np.round(value * (self.n_states - 1)) / (self.n_states - 1)
        return [quantized]

class DualReservoirDevice:
    def __init__(self, tau_volatile=3.64, n_nv_states=64):
        self.volatile = VolatileReservoir(tau=tau_volatile)
        self.non_volatile = NonVolatileMemory(n_states=n_nv_states)

class UltraReservoir:
    def __init__(self, device_profile=None, n_reservoir=1000, seed=42, spectral_radius=0.95, input_scale=0.3):
        np.random.seed(seed)
        self.n_reservoir = n_reservoir
        
        tau = 3.64
        n_nv_states = 64
        if device_profile:
            tau = device_profile.tau_volatile if device_profile.tau_volatile is not None else tau
            n_nv_states = device_profile.discrete_states_count if device_profile.discrete_states_count is not None else n_nv_states

        self.device = DualReservoirDevice(tau_volatile=tau, n_nv_states=n_nv_states)
        self.W_in = np.random.randn(n_reservoir, 2) * input_scale

        sparsity = 0.08
        W = np.random.randn(n_reservoir, n_reservoir) * 0.2
        mask = np.random.random((n_reservoir, n_reservoir)) < sparsity
        W = W * mask
        
        # Power iteration to find spectral radius (much faster than np.linalg.eigvals)
        b = np.random.randn(n_reservoir)
        for _ in range(15):
            W_b = W @ b
            b_norm = np.linalg.norm(W_b)
            if b_norm == 0:
                break
            b = W_b / b_norm
        max_eigen = np.linalg.norm(W @ b)
        
        self.W = W / max_eigen * spectral_radius if max_eigen > 0 else W
        self.taus = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    def extract_features(self, x, n_segments=10):
        flat = x.flatten()
        flat_norm = (flat - flat.min()) / (flat.max() - flat.min() + 1e-8)
        
        # 1. Simulate volatile and non-volatile responses once for the entire sequence
        v_states = []
        nv_states = []
        self.device.volatile.reset()
        for t in range(len(flat_norm)):
            x_norm = flat_norm[t]
            # Use actual current pixel input (x_norm) instead of a constant bias (0.1)
            v_state = self.device.volatile.get_response(x_norm)
            nv_state = self.device.non_volatile.encode(x_norm)[0]
            v_states.append(v_state)
            nv_states.append(nv_state)
        
        dual_inputs = np.stack([v_states, nv_states], axis=1) # (T, 2)
        
        # 2. Vectorized evolution over all tau scales
        tau_scale_col = np.array(self.taus)[:, np.newaxis] # (7, 1)
        state = np.zeros((len(self.taus), self.n_reservoir)) # (7, n_reservoir)
        
        states_all = []
        for t in range(len(flat_norm)):
            dual_input = dual_inputs[t]
            in_part = self.W_in @ dual_input
            pre_act = in_part + state @ self.W.T
            state = (1.0 - tau_scale_col) * state + tau_scale_col * np.tanh(pre_act)
            states_all.append(state.copy())
            
        states_all = np.array(states_all) # (T, 7, n_reservoir)
        
        all_features = []
        
        # 3. Fast feature statistics extraction using single-sort percentiles
        for j in range(len(self.taus)):
            states = states_all[:, j, :] # (T, n_reservoir)
            T = len(states)
            segment_size = max(1, T // n_segments)

            # Segmented statistics
            for i in range(n_segments):
                start = i * segment_size
                end = start + segment_size if i < n_segments - 1 else T
                seg = states[start:end]
                seg_flat = seg.flatten()
                seg_sorted = np.sort(seg_flat)
                L = len(seg_sorted)
                all_features.extend([
                    seg_sorted.mean(),
                    seg_sorted.std(),
                    seg_sorted[-1],
                    seg_sorted[0],
                    seg_sorted[L // 2],
                    seg_sorted[L // 4],
                    seg_sorted[(3 * L) // 4],
                ])

            # Global statistics
            states_flat = states.flatten()
            states_sorted = np.sort(states_flat)
            L_g = len(states_sorted)
            all_features.extend([
                states_sorted.mean(),
                states_sorted.std(),
                states_sorted[-1],
                states_sorted[0],
                states[-1].mean(),
                states[-1].std(),
            ])

            # Difference features
            diff = np.diff(states, axis=0)
            diff_flat = diff.flatten()
            diff_sorted = np.sort(diff_flat)
            all_features.extend([
                diff_sorted.mean(),
                diff_sorted.std(),
                diff_sorted[-1],
                diff_sorted[0],
            ])

        return np.array(all_features)
