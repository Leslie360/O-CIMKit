import torch
import torch.nn as nn
import numpy as np
from core.physics import inject_gaussian_noise, apply_non_linear_gradient, simulate_conductance_drift
from core.quantization import LSQQuantizer, MinMaxQuantizer, STEClamp

class OrganicSynapseConv(nn.Conv2d):
    """
    Hardware-Aware Convolutional Layer mapping mathematical weights to
    physical device conductances and simulating C2C write noise and LTP/LTD gradient non-linearities.
    """
    def __init__(self, in_channels, out_channels, kernel_size, device_profile, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.profile = device_profile
        self.drift_hours = 0.0
        
        if self.profile is not None:
            self.phys_min = self.profile.conductance_min
            self.phys_max = self.profile.conductance_max
            self.noise_std = self.profile.get_noise_std()
            
            # Register backward hook to modify gradients based on LTP/LTD non-linear slopes
            if self.profile.ltp_poly_coefficients and self.profile.ltd_poly_coefficients:
                self.weight.register_hook(
                    lambda grad: apply_non_linear_gradient(
                        grad, self.weight, 
                        self.profile.ltp_poly_coefficients, 
                        self.profile.ltd_poly_coefficients
                    )
                )
        else:
            self.phys_min = None
            self.phys_max = None
            self.noise_std = 0.0

    def forward(self, input):
        if self.profile is None:
            return self._conv_forward(input, self.weight, self.bias)
            
        # 1. Map mathematical weights [-1, 1] to physical conductance [G_min, G_max]
        w_clamp = STEClamp.apply(self.weight, -1.0, 1.0)
        w_norm = (w_clamp + 1.0) / 2.0
        w_phys = w_norm * (self.phys_max - self.phys_min) + self.phys_min
        
        # 2. Inject noise (training only)
        if self.training and self.noise_std > 0:
            w_phys = inject_gaussian_noise(w_phys, self.noise_std)
            
        # 2b. Apply physical conductance drift over time (aging)
        factor = 1.0
        if self.drift_hours > 0:
            drift_exp = getattr(self.profile, 'drift_exponent', 0.06)
            ret_noise = getattr(self.profile, 'retention_noise', 0.0005)
            if self.drift_hours > 1.0:
                factor = (self.drift_hours / 1.0) ** (-drift_exp)
                w_phys = w_phys * factor
                if ret_noise > 0:
                    noise_std_abs = ret_noise * (self.phys_max - self.phys_min) * (self.drift_hours ** 0.15)
                    w_phys = w_phys + torch.randn_like(w_phys) * noise_std_abs
                w_phys = torch.clamp(w_phys, min=self.phys_min * factor, max=self.phys_max * factor)
            
        # 3. Map noisy physical conductance back to mathematical domain [-1, 1]
        phys_min_drift = self.phys_min * factor
        phys_max_drift = self.phys_max * factor
        w_norm_noisy = (w_phys - phys_min_drift) / (phys_max_drift - phys_min_drift + 1e-20)
        w_math_noisy = w_norm_noisy * 2.0 - 1.0
        
        # 4. Perform standard convolution with modified weights
        return self._conv_forward(input, w_math_noisy, self.bias)

class QATMLPLayer(nn.Linear):
    """
    Quantization-Aware Training (QAT) Fully Connected Layer.
    Quantizes weights to discrete states from DeviceProfile using LSQ or MinMax.
    """
    def __init__(self, in_features, out_features, device_profile=None, bias=True, mode="minmax"):
        super().__init__(in_features, out_features, bias=bias)
        self.profile = device_profile
        self.quantizer = None
        self.drift_hours = 0.0
        
        if self.profile and self.profile.discrete_states_count is not None:
            if mode == "minmax":
                self.quantizer = MinMaxQuantizer(num_states=self.profile.discrete_states_count)
            else:
                self.quantizer = LSQQuantizer(num_states=self.profile.discrete_states_count)
 
    def forward(self, input):
        w = self.weight
        if self.quantizer is not None:
            # Apply Learned Step Size Quantization (LSQ) on weights during forward pass
            w = self.quantizer(self.weight)
            
        if self.profile is not None:
            phys_min = self.profile.conductance_min
            phys_max = self.profile.conductance_max
            w_clamp = STEClamp.apply(w, -1.0, 1.0)
            w_norm = (w_clamp + 1.0) / 2.0
            w_phys = w_norm * (phys_max - phys_min) + phys_min
            
            factor = 1.0
            if self.drift_hours > 0:
                drift_exp = getattr(self.profile, 'drift_exponent', 0.06)
                ret_noise = getattr(self.profile, 'retention_noise', 0.0005)
                if self.drift_hours > 1.0:
                    factor = (self.drift_hours / 1.0) ** (-drift_exp)
                    w_phys = w_phys * factor
                    if ret_noise > 0:
                        noise_std_abs = ret_noise * (phys_max - phys_min) * (self.drift_hours ** 0.15)
                        w_phys = w_phys + torch.randn_like(w_phys) * noise_std_abs
                    w_phys = torch.clamp(w_phys, min=phys_min * factor, max=phys_max * factor)
                    
            phys_min_drift = phys_min * factor
            phys_max_drift = phys_max * factor
            w_norm_drift = (w_phys - phys_min_drift) / (phys_max_drift - phys_min_drift + 1e-20)
            w = w_norm_drift * 2.0 - 1.0
            
        return nn.functional.linear(input, w, self.bias)

class PhysicalReservoir:
    """
    Physical Reservoir Computing Layer using device volatile/non-volatile time-scales.
    """
    def __init__(self, n_inputs, n_reservoir, device_profile, dual_scale=True, seed=42, device='cpu'):
        self.n_reservoir = n_reservoir
        self.profile = device_profile
        self.device = device
        self.dual_scale = dual_scale
        
        # Get volatile leaking rate
        if self.profile.is_volatile:
            self.alpha_fast = self.profile.leaking_rate_volatile
        else:
            self.alpha_fast = 0.24  # Default fast leaking rate
            
        # Define slow time scale for dual scale
        self.alpha_slow = 0.027  # Default slow leaking rate
        
        torch.manual_seed(seed)
        
        if self.dual_scale:
            self.n_fast = n_reservoir // 2
            self.n_slow = n_reservoir - self.n_fast
            
            # Fast scale pathway
            self.W_in_fast = torch.randn(self.n_fast, n_inputs, device=device) * 0.5
            W_f = torch.randn(self.n_fast, self.n_fast, device=device) * 0.1
            mask_f = torch.rand(self.n_fast, self.n_fast, device=device) < 0.1
            self.W_fast = self._scale_spectral_radius(W_f * mask_f)
            
            # Slow scale pathway
            self.W_in_slow = torch.randn(self.n_slow, n_inputs, device=device) * 0.5
            W_s = torch.randn(self.n_slow, self.n_slow, device=device) * 0.1
            mask_s = torch.rand(self.n_slow, self.n_slow, device=device) < 0.1
            self.W_slow = self._scale_spectral_radius(W_s * mask_s)
        else:
            self.W_in = torch.randn(n_reservoir, n_inputs, device=device) * 0.5
            W = torch.randn(n_reservoir, n_reservoir, device=device) * 0.1
            mask = torch.rand(n_reservoir, n_reservoir, device=device) < 0.1
            self.W = self._scale_spectral_radius(W * mask)

    def _scale_spectral_radius(self, W, target=0.9):
        """Scale spectral radius of reservoir matrix to preserve echo state property."""
        if W.shape[0] == 0:
            return W
        try:
            eig_vals = torch.linalg.eigvals(W)
            max_eig = eig_vals.abs().max()
            if max_eig > 0:
                W = W / max_eig * target
        except Exception:
            pass
        return W

    def process_sequence(self, X):
        """
        Process an input sequence of length T and return the reservoir state history.
        Args:
            X (np.ndarray or torch.Tensor): Input sequence shape (T, n_inputs).
        Returns:
            np.ndarray: Unified state representation shape (T, n_reservoir).
        """
        T = X.shape[0]
        states = []
        
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(self.device)
            
        if self.dual_scale:
            state_fast = torch.zeros(self.n_fast, device=self.device)
            state_slow = torch.zeros(self.n_slow, device=self.device)
            
            for t in range(T):
                u = X[t]
                
                # Fast update
                pre_fast = self.W_in_fast @ u + self.W_fast @ state_fast
                state_fast = (1.0 - self.alpha_fast) * state_fast + self.alpha_fast * torch.tanh(pre_fast)
                
                # Slow update
                pre_slow = self.W_in_slow @ u + self.W_slow @ state_slow
                state_slow = (1.0 - self.alpha_slow) * state_slow + self.alpha_slow * torch.tanh(pre_slow)
                
                # Concatenate fast and slow states
                state = torch.cat([state_fast, state_slow])
                states.append(state.cpu().numpy())
        else:
            state = torch.zeros(self.n_reservoir, device=self.device)
            for t in range(T):
                u = X[t]
                pre = self.W_in @ u + self.W @ state
                state = (1.0 - self.alpha_fast) * state + self.alpha_fast * torch.tanh(pre)
                states.append(state.cpu().numpy())
                
        return np.array(states)

    def extract_temporal_features(self, state_history):
        """
        Extract high-dimensional features by concatenating temporal stats.
        Concats: Mean, Std, Max, Min, and final state.
        Shape output: (5 * n_reservoir,)
        """
        # state_history shape: (T, n_reservoir)
        mean = np.mean(state_history, axis=0)
        std = np.std(state_history, axis=0)
        max_v = np.max(state_history, axis=0)
        min_v = np.min(state_history, axis=0)
        last = state_history[-1]
        
        return np.concatenate([mean, std, max_v, min_v, last])

class DynamicOrganicSynapse(nn.Module):
    """
    Dynamic Organic Synapse Layer unifying volatile relaxation (short-term memory)
    and non-volatile quantized weight mapping (long-term memory) in a single physical layer.
    Processes sequences and maintains temporal state dynamics.
    """
    def __init__(self, in_features, out_features, device_profile=None, bias=True, mode="minmax"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.profile = device_profile
        self.quantizer = None
        self.drift_hours = 0.0
        
        # Learnable weight and bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
        if self.profile and self.profile.discrete_states_count is not None:
            if mode == "minmax":
                self.quantizer = MinMaxQuantizer(num_states=self.profile.discrete_states_count)
            else:
                self.quantizer = LSQQuantizer(num_states=self.profile.discrete_states_count)
                
        # Leaking rate (volatile dynamic property)
        if self.profile and self.profile.is_volatile:
            self.alpha = self.profile.leaking_rate_volatile
        else:
            self.alpha = 0.35 # Default leaking rate for short-term memory

    def forward(self, X, return_sequence=False):
        """
        Args:
            X (torch.Tensor): Input sequence shape (batch_size, seq_len, in_features)
            return_sequence (bool): If True, returns full state history, else final state.
        """
        batch_size, seq_len, _ = X.shape
        device = X.device
        
        # 1. Map weights to hardware constraints
        w = self.weight
        if self.quantizer is not None:
            w = self.quantizer(self.weight)
            
        if self.profile is not None:
            phys_min = self.profile.conductance_min
            phys_max = self.profile.conductance_max
            w_clamp = STEClamp.apply(w, -1.0, 1.0)
            w_norm = (w_clamp + 1.0) / 2.0
            w_phys = w_norm * (phys_max - phys_min) + phys_min
            
            factor = 1.0
            if self.drift_hours > 0:
                drift_exp = getattr(self.profile, 'drift_exponent', 0.06)
                ret_noise = getattr(self.profile, 'retention_noise', 0.0005)
                if self.drift_hours > 1.0:
                    factor = (self.drift_hours / 1.0) ** (-drift_exp)
                    w_phys = w_phys * factor
                    if ret_noise > 0:
                        noise_std_abs = ret_noise * (phys_max - phys_min) * (self.drift_hours ** 0.15)
                        w_phys = w_phys + torch.randn_like(w_phys) * noise_std_abs
                    w_phys = torch.clamp(w_phys, min=phys_min * factor, max=phys_max * factor)
                    
            phys_min_drift = phys_min * factor
            phys_max_drift = phys_max * factor
            w_norm_drift = (w_phys - phys_min_drift) / (phys_max_drift - phys_min_drift + 1e-20)
            w = w_norm_drift * 2.0 - 1.0
            
        # 2. Iterate over time sequence
        h = torch.zeros(batch_size, self.out_features, device=device)
        states = []
        
        for t in range(seq_len):
            u = X[:, t, :]
            pre = nn.functional.linear(u, w, self.bias)
            if self.profile is not None and self.drift_hours > 0:
                pre = pre * factor
            h = (1.0 - self.alpha) * h + self.alpha * torch.tanh(pre)
            if return_sequence:
                states.append(h.clone().unsqueeze(1))
                
        if return_sequence:
            return torch.cat(states, dim=1)
        return h

class SelfHealingCrossbar(nn.Linear):
    """
    On-Chip Unsupervised Self-Healing Crossbar Layer.
    Supports multiple dynamic compensation modes to mitigate device drift and noise:
    1. 'none': Naive uncompensated drift.
    2. 'global_scaling': IBM AIHWKit-style global scaling correction.
    3. 'reference_calibration': On-chip reference column calibration (with process variation).
    4. 'self_healing': Unsupervised online mean & variance alignment.
    """
    def __init__(self, in_features, out_features, device_profile=None, bias=True, mode="minmax"):
        super().__init__(in_features, out_features, bias=bias)
        self.profile = device_profile
        self.quantizer = None
        self.drift_hours = 0.0
        self.self_healing_enabled = True
        self.compensation_mode = "self_healing"
        
        if self.profile and self.profile.discrete_states_count is not None:
            if mode == "minmax":
                self.quantizer = MinMaxQuantizer(num_states=self.profile.discrete_states_count)
            else:
                self.quantizer = LSQQuantizer(num_states=self.profile.discrete_states_count)
                
        # Register baseline activation statistics
        self.register_buffer("baseline_mean", torch.zeros(out_features))
        self.register_buffer("baseline_var", torch.ones(out_features))
        
        # Online running statistics tracking
        self.register_buffer("running_mean", torch.zeros(out_features))
        self.register_buffer("running_var", torch.ones(out_features))
        self.momentum = 0.1
        self.epsilon = 1e-5
        self.is_baseline_calibrated = False

    def forward(self, input):
        # 1. Apply QAT and hardware constraints (with power-law drift)
        w = self.weight
        if self.quantizer is not None:
            w = self.quantizer(self.weight)
            
        if self.profile is not None:
            phys_min = self.profile.conductance_min
            phys_max = self.profile.conductance_max
            w_clamp = STEClamp.apply(w, -1.0, 1.0)
            w_norm = (w_clamp + 1.0) / 2.0
            w_phys = w_norm * (phys_max - phys_min) + phys_min
            
            # Column-wise (channel-wise) drift exponent D2D variation
            nominal_drift_exp = getattr(self.profile, 'drift_exponent', 0.06)
            # Use deterministic seeding based on shape to keep it reproducible
            rng = torch.Generator(device=w.device).manual_seed(self.out_features)
            drift_exp = torch.rand(self.out_features, 1, generator=rng, device=w.device) * 0.04 + (nominal_drift_exp - 0.02)
            drift_exp = torch.clamp(drift_exp, min=0.01)
            
            factor = torch.ones(self.out_features, 1, device=w.device)
            if self.drift_hours > 0:
                ret_noise = getattr(self.profile, 'retention_noise', 0.0005)
                if self.drift_hours > 1.0:
                    factor = (self.drift_hours / 1.0) ** (-drift_exp)
                    w_phys = w_phys * factor
                    if ret_noise > 0:
                        noise_std_abs = ret_noise * (phys_max - phys_min) * (self.drift_hours ** 0.15)
                        w_phys = w_phys + torch.randn_like(w_phys) * noise_std_abs
                    w_phys = torch.clamp(w_phys, min=phys_min * factor, max=phys_max * factor)
                    
            phys_min_drift = phys_min * factor
            phys_max_drift = phys_max * factor
            w_norm_drift = (w_phys - phys_min_drift) / (phys_max_drift - phys_min_drift + 1e-20)
            w = w_norm_drift * 2.0 - 1.0
            
        # 2. Linear projection
        out = nn.functional.linear(input, w, self.bias)
        
        # Apply physical current decay under drift before compensation
        if self.profile is not None and not self.training and self.drift_hours > 0:
            out = out * factor.squeeze()
            
            # Inject dynamic read/inference noise (5% of output standard deviation)
            read_noise = torch.randn_like(out) * 0.05 * (out.std() + 1e-8)
            out = out + read_noise
        
        # 3. Stats calibration / Self-healing
        if self.training:
            # During training, track baseline stats using batch statistics
            with torch.no_grad():
                if out.dim() > 2:
                    flat_out = out.view(-1, out.size(-1))
                else:
                    flat_out = out
                batch_mean = flat_out.mean(dim=0)
                batch_var = flat_out.var(dim=0, unbiased=False)
                
                # Update baseline buffers
                self.baseline_mean.copy_(self.baseline_mean * (1 - self.momentum) + batch_mean * self.momentum)
                self.baseline_var.copy_(self.baseline_var * (1 - self.momentum) + batch_var * self.momentum)
                self.is_baseline_calibrated = True
                
                # Sync running statistics to baseline initially
                self.running_mean.copy_(self.baseline_mean)
                self.running_var.copy_(self.baseline_var)
        else:
            comp_mode = getattr(self, "compensation_mode", "self_healing")
            if not self.self_healing_enabled:
                comp_mode = "none"
                
            if comp_mode == "none":
                pass
            elif comp_mode == "global_scaling":
                # Scale by 1/factor to revert nominal decay (using nominal factor)
                nominal_drift_exp = getattr(self.profile, 'drift_exponent', 0.06)
                nominal_factor = (self.drift_hours / 1.0) ** (-nominal_drift_exp) if self.drift_hours > 1.0 else 1.0
                out = out / (nominal_factor + 1e-20)
            elif comp_mode == "reference_calibration":
                # Simulated reference column tracking with process variation (3% read noise)
                with torch.no_grad():
                    nominal_drift_exp = getattr(self.profile, 'drift_exponent', 0.06)
                    nominal_factor = (self.drift_hours / 1.0) ** (-nominal_drift_exp) if self.drift_hours > 1.0 else 1.0
                    ref_noise = torch.randn_like(self.baseline_mean) * 0.03
                    ref_factor = nominal_factor * (1.0 + ref_noise)
                out = out / (ref_factor + 1e-20)
            elif comp_mode == "self_healing" and self.is_baseline_calibrated:
                # Update running statistics of degraded crossbar outputs
                with torch.no_grad():
                    if out.dim() > 2:
                        flat_out = out.view(-1, out.size(-1))
                    else:
                        flat_out = out
                    eval_mean = flat_out.mean(dim=0)
                    eval_var = flat_out.var(dim=0, unbiased=False)
                    
                    # Track online statistics using exponential moving average
                    self.running_mean.copy_(self.running_mean * (1 - self.momentum) + eval_mean * self.momentum)
                    self.running_var.copy_(self.running_var * (1 - self.momentum) + eval_var * self.momentum)
                
                # Determine whether to use batch stats or running stats
                use_batch_stats = flat_out.size(0) > 4
                comp_mean = eval_mean if use_batch_stats else self.running_mean
                comp_var = eval_var if use_batch_stats else self.running_var
                
                # Compute scaling and shift compensation factors
                beta = torch.sqrt(self.baseline_var) / (torch.sqrt(comp_var) + self.epsilon)
                gamma = self.baseline_mean - beta * comp_mean
                
                # Apply compensation to output activations
                out = out * beta + gamma
                
        return out


class SelfHealingConv2d(OrganicSynapseConv):
    """
    On-Chip Unsupervised Self-Healing Convolutional Layer.
    Tracks channel-wise activation statistics across batches/spatial grids and compensates for drift.
    Supports multiple dynamic compensation modes: 'none', 'global_scaling', 'reference_calibration', 'self_healing'.
    """
    def __init__(self, in_channels, out_channels, kernel_size, device_profile, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, device_profile, **kwargs)
        self.self_healing_enabled = True
        self.compensation_mode = "self_healing"
        
        # Register buffers for channel-wise activation statistics
        self.register_buffer("baseline_mean", torch.zeros(out_channels))
        self.register_buffer("baseline_var", torch.ones(out_channels))
        
        self.register_buffer("running_mean", torch.zeros(out_channels))
        self.register_buffer("running_var", torch.ones(out_channels))
        self.momentum = 0.1
        self.epsilon = 1e-5
        self.is_baseline_calibrated = False

    def forward(self, input):
        factor = torch.ones(self.out_channels, 1, 1, 1, device=self.weight.device)
        if self.profile is not None:
            # Channel-wise drift exponent variation
            nominal_drift_exp = getattr(self.profile, 'drift_exponent', 0.06)
            rng = torch.Generator(device=self.weight.device).manual_seed(self.out_channels)
            drift_exp = torch.rand(self.out_channels, 1, 1, 1, generator=rng, device=self.weight.device) * 0.04 + (nominal_drift_exp - 0.02)
            drift_exp = torch.clamp(drift_exp, min=0.01)
            
            if self.drift_hours > 1.0:
                factor = (self.drift_hours / 1.0) ** (-drift_exp)
                
        # Temporarily set OrganicSynapseConv's drift_hours to 0 so it doesn't double-apply drift
        old_drift = self.drift_hours
        self.drift_hours = 0.0
        
        # We manually apply drift to weights here for channel-wise precision
        w = self.weight
        phys_min = self.profile.conductance_min
        phys_max = self.profile.conductance_max
        w_clamp = STEClamp.apply(w, -1.0, 1.0)
        w_norm = (w_clamp + 1.0) / 2.0
        w_phys = w_norm * (phys_max - phys_min) + phys_min
        
        if old_drift > 1.0:
            w_phys = w_phys * factor
            ret_noise = getattr(self.profile, 'retention_noise', 0.0005)
            if ret_noise > 0:
                noise_std_abs = ret_noise * (phys_max - phys_min) * (old_drift ** 0.15)
                w_phys = w_phys + torch.randn_like(w_phys) * noise_std_abs
            w_phys = torch.clamp(w_phys, min=phys_min * factor, max=phys_max * factor)
            
        phys_min_drift = phys_min * factor
        phys_max_drift = phys_max * factor
        w_norm_drift = (w_phys - phys_min_drift) / (phys_max_drift - phys_min_drift + 1e-20)
        w_math_noisy = w_norm_drift * 2.0 - 1.0
        
        # Run convolution
        out = self._conv_forward(input, w_math_noisy, self.bias)
        
        # Restore drift_hours
        self.drift_hours = old_drift
        
        # Apply physical current decay under drift before compensation
        if self.profile is not None and not self.training and self.drift_hours > 0:
            out = out * factor.view(1, -1, 1, 1)
            
            # Inject dynamic read/inference noise (5% of output standard deviation)
            read_noise = torch.randn_like(out) * 0.05 * (out.std() + 1e-8)
            out = out + read_noise
            
        # Stats calibration / Self-healing
        if self.training:
            with torch.no_grad():
                # Shape: [B, C, H, W] -> permute to [B, H, W, C] -> flat to [-1, C]
                flat_out = out.permute(0, 2, 3, 1).reshape(-1, out.size(1))
                batch_mean = flat_out.mean(dim=0)
                batch_var = flat_out.var(dim=0, unbiased=False)
                
                # Update baseline buffers
                self.baseline_mean.copy_(self.baseline_mean * (1 - self.momentum) + batch_mean * self.momentum)
                self.baseline_var.copy_(self.baseline_var * (1 - self.momentum) + batch_var * self.momentum)
                self.is_baseline_calibrated = True
                
                # Initialize running statistics
                self.running_mean.copy_(self.baseline_mean)
                self.running_var.copy_(self.baseline_var)
        else:
            comp_mode = getattr(self, "compensation_mode", "self_healing")
            if not self.self_healing_enabled:
                comp_mode = "none"
                
            if comp_mode == "none":
                pass
            elif comp_mode == "global_scaling":
                nominal_drift_exp = getattr(self.profile, 'drift_exponent', 0.06)
                nominal_factor = (self.drift_hours / 1.0) ** (-nominal_drift_exp) if self.drift_hours > 1.0 else 1.0
                out = out / (nominal_factor + 1e-20)
            elif comp_mode == "reference_calibration":
                with torch.no_grad():
                    nominal_drift_exp = getattr(self.profile, 'drift_exponent', 0.06)
                    nominal_factor = (self.drift_hours / 1.0) ** (-nominal_drift_exp) if self.drift_hours > 1.0 else 1.0
                    ref_noise = torch.randn_like(self.baseline_mean) * 0.03
                    ref_factor = nominal_factor * (1.0 + ref_noise)
                out = out / (ref_factor.view(1, -1, 1, 1) + 1e-20)
            elif comp_mode == "self_healing" and self.is_baseline_calibrated:
                with torch.no_grad():
                    flat_out = out.permute(0, 2, 3, 1).reshape(-1, out.size(1))
                    eval_mean = flat_out.mean(dim=0)
                    eval_var = flat_out.var(dim=0, unbiased=False)
                    
                    self.running_mean.copy_(self.running_mean * (1 - self.momentum) + eval_mean * self.momentum)
                    self.running_var.copy_(self.running_var * (1 - self.momentum) + eval_var * self.momentum)
                    
                # Use batch stats if flattened batch size is large enough
                use_batch_stats = flat_out.size(0) > 64
                comp_mean = eval_mean if use_batch_stats else self.running_mean
                comp_var = eval_var if use_batch_stats else self.running_var
                
                beta = (torch.sqrt(self.baseline_var) / (torch.sqrt(comp_var) + self.epsilon)).view(1, -1, 1, 1)
                gamma = (self.baseline_mean - beta.squeeze() * comp_mean).view(1, -1, 1, 1)
                
                out = out * beta + gamma
                
        return out


