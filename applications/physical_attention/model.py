import numpy as np

class PhysicalKVCacheFinal:
    """
    Physical KV-Cache device model.
    - Query (Q): Volatile state x_v(t) = alpha_v * x_v(t-1) + (1-alpha_v) * input(t)
    - Key (K) & Value (V): Non-volatile statistical features of the sequence.
    - Output: softmax(Q * K^T) * V
    """
    def __init__(self, device_profile=None, tau_volatile=3.64, n_states=64):
        self.profile = device_profile
        
        if self.profile:
            tau_volatile = self.profile.tau_volatile if self.profile.tau_volatile is not None else tau_volatile
            n_states = self.profile.discrete_states_count if self.profile.discrete_states_count is not None else n_states
            
        self.tau_volatile = tau_volatile
        self.n_states = n_states
        self.alpha_v = np.exp(-1.0 / tau_volatile)
        self.x_v = 0.0
        self.x_nv = np.zeros(n_states)

    def reset(self):
        self.x_v = 0.0

    def write_nv_statistics(self, sequence):
        stats = [
            np.mean(sequence),
            np.std(sequence),
            np.max(sequence),
            np.min(sequence),
            np.percentile(sequence, 25),
            np.percentile(sequence, 75),
            np.mean(np.diff(sequence)),
            np.std(np.diff(sequence))
        ]
        
        # Quantize and store
        for i, stat in enumerate(stats[:self.n_states]):
            quantized = np.round(stat * 100.0) / 100.0
            self.x_nv[i] = quantized

    def step(self, input_t):
        self.x_v = self.alpha_v * self.x_v + (1.0 - self.alpha_v) * input_t
        k = self.x_nv.copy()
        v = self.x_nv.copy()

        # Attention weights: softmax(Q * K^T)
        attention_scores = self.x_v * k
        exp_scores = np.exp(attention_scores - np.max(attention_scores))
        attention_weights = exp_scores / (np.sum(exp_scores) + 1e-8)

        output = np.sum(attention_weights * v)
        return self.x_v, output, attention_weights

    def process_sequence(self, sequence):
        self.write_nv_statistics(sequence)
        self.reset()

        n_steps = len(sequence)
        queries = np.zeros(n_steps)
        outputs = np.zeros(n_steps)
        attention_weights = np.zeros((n_steps, self.n_states))

        for t in range(n_steps):
            queries[t], outputs[t], attention_weights[t] = self.step(sequence[t])

        return queries, outputs, attention_weights

class VolatileOnlyModel:
    def __init__(self, tau=3.64):
        self.alpha = np.exp(-1.0 / tau)
        self.state = 0.0

    def reset(self):
        self.state = 0.0

    def process_sequence(self, sequence):
        self.reset()
        n_steps = len(sequence)
        states = np.zeros(n_steps)

        for t in range(n_steps):
            self.state = self.alpha * self.state + (1.0 - self.alpha) * sequence[t]
            states[t] = self.state

        return states

class NonVolatileOnlyModel:
    def __init__(self, n_states=64):
        self.n_states = n_states
        self.states = np.zeros(n_states)

    def process_sequence(self, sequence):
        stats = [
            np.mean(sequence),
            np.std(sequence),
            np.max(sequence),
            np.min(sequence),
            np.percentile(sequence, 25),
            np.percentile(sequence, 75),
            np.mean(np.diff(sequence)),
            np.std(np.diff(sequence))
        ]

        for i, stat in enumerate(stats[:self.n_states]):
            self.states[i] = stat

        return self.states.copy()
