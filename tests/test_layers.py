import unittest
import torch
import numpy as np
from core.layers import DynamicOrganicSynapse, SelfHealingCrossbar
from profiles.device_profile import DeviceProfile

class TestLayers(unittest.TestCase):
    def get_mock_profile(self):
        return DeviceProfile(
            name="TestDevice",
            device_type="memristor",
            is_volatile=True,
            tau_volatile=1.0,
            conductance_min=1e-11,
            conductance_max=1e-10,
            noise_std_ratio=0.01
        )

    def test_dynamic_organic_synapse(self):
        profile = self.get_mock_profile()
        layer = DynamicOrganicSynapse(in_features=10, out_features=5, device_profile=profile)
        x = torch.randn(2, 15, 10)
        out = layer(x, return_sequence=False)
        self.assertEqual(out.shape, (2, 5))
        
        out_seq = layer(x, return_sequence=True)
        self.assertEqual(out_seq.shape, (2, 15, 5))

    def test_self_healing_crossbar(self):
        profile = self.get_mock_profile()
        profile.is_volatile = False
        profile.discrete_states_count = 16
        
        layer = SelfHealingCrossbar(in_features=8, out_features=4, device_profile=profile)
        
        # 1. Training (Baseline calibration)
        layer.train()
        x_train = torch.randn(10, 8)
        for _ in range(5):
            out = layer(x_train)
        self.assertTrue(layer.is_baseline_calibrated)
        
        # 2. Evaluation with different compensation modes under drift
        layer.eval()
        layer.drift_hours = 100.0
        
        for mode in ["none", "global_scaling", "reference_calibration", "self_healing"]:
            layer.compensation_mode = mode
            out_eval = layer(x_train)
            self.assertEqual(out_eval.shape, (10, 4))
            
    def test_self_healing_conv2d(self):
        from core.layers import SelfHealingConv2d
        profile = self.get_mock_profile()
        profile.is_volatile = False
        profile.discrete_states_count = 16
        
        # Conv2d layer
        layer = SelfHealingConv2d(
            in_channels=3, out_channels=8, kernel_size=3, device_profile=profile, padding=1
        )
        
        # 1. Training
        layer.train()
        x_train = torch.randn(4, 3, 16, 16)
        for _ in range(5):
            out = layer(x_train)
        self.assertTrue(layer.is_baseline_calibrated)
        
        # 2. Evaluation
        layer.eval()
        layer.drift_hours = 100.0
        
        for mode in ["none", "global_scaling", "reference_calibration", "self_healing"]:
            layer.compensation_mode = mode
            out_eval = layer(x_train)
            self.assertEqual(out_eval.shape, (4, 8, 16, 16))

