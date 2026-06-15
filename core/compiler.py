import os
import sys
import json
import torch
import torch.nn as nn

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiles.device_profile import DeviceProfile
from core.layers import QATMLPLayer, SelfHealingCrossbar, DynamicOrganicSynapse

class BionicCoDesignCompiler:
    """
    CoDesign Compiler automatically translates physical device profile parameters 
    into optimal hardware-aware neural network configurations, quantizer modes, and training hyperparameters.
    """
    def __init__(self, device_profile_path):
        self.profile_path = device_profile_path
        self.profile = DeviceProfile.from_json(device_profile_path)
        
    def compile(self):
        """
        Analyzes device characteristics and compiles compilation guidelines.
        Returns:
            dict: Compiled recommendations.
        """
        print(f"⚙️ Compiling co-design settings for device profile: {self.profile.device_name}...")
        
        # 1. Analyze Quantization & States
        num_states = self.profile.discrete_states_count
        if num_states is None:
            quantizer_mode = "ideal"
        elif num_states >= 64:
            quantizer_mode = "lsq" # High state density benefits from learned step size QAT
        else:
            quantizer_mode = "minmax" # Low state density benefits from hard min/max clamping
            
        # 2. Analyze Noise & Training Stability
        c2c_noise = self.profile.get_noise_std()
        if c2c_noise > 0.05:
            suggested_lr = 0.0002 # High noise requires conservative learning rate
            suggested_wd = 0.005  # Heavy weight decay helps smooth gradients
            epochs_multiplier = 1.5 # Needs longer training to converge
        else:
            suggested_lr = 0.002
            suggested_wd = 0.001
            epochs_multiplier = 1.0
            
        # 3. Analyze Volatility & Dynamics
        is_volatile = self.profile.is_volatile
        if is_volatile:
            tau = self.profile.tau_volatile if self.profile.tau_volatile is not None else 3.64
            leaking_rate = self.profile.leaking_rate_volatile if self.profile.leaking_rate_volatile is not None else 0.35
            layer_recommendation = "DynamicOrganicSynapse"
        else:
            tau = "N/A"
            leaking_rate = "N/A"
            layer_recommendation = "SelfHealingCrossbar"
            
        # 4. Analyze Drift & Reliability
        drift_exp = getattr(self.profile, 'drift_exponent', 0.06)
        if drift_exp > 0.08:
            healing_policy = "unsupervised_active" # High drift requires online self-healing
        else:
            healing_policy = "reference_calibration" # Moderate drift can be resolved by periodic ref-cell readouts
            
        compiled_dict = {
            "device_name": self.profile.device_name,
            "device_type": "Volatile" if is_volatile else "Non-Volatile",
            "quantizer_mode": quantizer_mode,
            "num_states": num_states,
            "suggested_lr": suggested_lr,
            "suggested_wd": suggested_wd,
            "epochs_multiplier": epochs_multiplier,
            "layer_recommendation": layer_recommendation,
            "leaking_rate": leaking_rate,
            "healing_policy": healing_policy,
            "drift_exponent": drift_exp
        }
        
        # Display compilation report
        print("=" * 60)
        print("🛠️ CO-DESIGN COMPILER REPORT")
        print("=" * 60)
        print(f"  Device Name:            {compiled_dict['device_name']}")
        print(f"  Memory Category:        {compiled_dict['device_type']}")
        print(f"  Quantization Strategy:  {compiled_dict['quantizer_mode'].upper()} QAT ({num_states} states)")
        print(f"  Suggested Learning Rate: {compiled_dict['suggested_lr']}")
        print(f"  Suggested Weight Decay:  {compiled_dict['suggested_wd']}")
        print(f"  Dynamic Layer Spec:     {compiled_dict['layer_recommendation']} (Leaking Rate: {leaking_rate})")
        print(f"  Reliability Policy:     {compiled_dict['healing_policy'].replace('_', ' ').upper()} (Drift Exp: {drift_exp})")
        print("=" * 60)
        
        return compiled_dict

    def synthesize_model(self, model_type, in_features, out_features, hidden_dim=256):
        """
        Dynamically synthesizes a PyTorch model mapping to the compiled physical properties.
        """
        compiled = self.compile()
        
        if model_type == "sequential":
            # For time-series/sequential input, synthesize a DynamicOrganicSynapse pipeline
            class CompiledSequentialModel(nn.Module):
                def __init__(self, in_dim, out_dim, h_dim, profile, q_mode):
                    super().__init__()
                    self.dynamic_synapse = DynamicOrganicSynapse(
                        in_dim, h_dim, device_profile=profile, mode=q_mode
                    )
                    self.readout = SelfHealingCrossbar(
                        h_dim, out_dim, device_profile=profile, mode=q_mode
                    )
                    
                def forward(self, x):
                    # x shape: (batch_size, seq_len, in_dim)
                    h = self.dynamic_synapse(x, return_sequence=False)
                    return self.readout(h)
                    
            model = CompiledSequentialModel(
                in_features, out_features, hidden_dim, self.profile, compiled["quantizer_mode"]
            )
            print("🚀 Synthesized Dynamic Sequential Model Pipeline (Uni-Synapse STP + LTP Readout)")
            return model
            
        else:
            # For standard feedforward classifier, synthesize a SelfHealingCrossbar MLP pipeline
            class CompiledMLPModel(nn.Module):
                def __init__(self, in_dim, out_dim, h_dim, profile, q_mode):
                    super().__init__()
                    self.layer1 = SelfHealingCrossbar(in_dim, h_dim, device_profile=profile, mode=q_mode)
                    self.layer2 = SelfHealingCrossbar(h_dim, out_dim, device_profile=profile, mode=q_mode)
                    self.relu = nn.ReLU()
                    
                def forward(self, x):
                    h = self.relu(self.layer1(x))
                    return self.layer2(h)
                    
            model = CompiledMLPModel(
                in_features, out_features, hidden_dim, self.profile, compiled["quantizer_mode"]
            )
            print("🚀 Synthesized Feedforward Self-Healing Crossbar MLP Pipeline")
            return model
