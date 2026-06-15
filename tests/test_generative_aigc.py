import unittest
import torch
import numpy as np
from profiles.device_profile import DeviceProfile
from applications.generative_aigc.model import ConvVAE
from applications.generative_aigc.train import convert_to_self_healing, vae_loss_fn
from core.layers import SelfHealingCrossbar, SelfHealingConv2d

class TestGenerativeAIGC(unittest.TestCase):
    def setUp(self):
        # Create a mock device profile
        self.profile = DeviceProfile(
            name="TestMemristor",
            device_type="memristor",
            is_volatile=False,
            conductance_min=1e-6,
            conductance_max=1e-4,
            noise_std_ratio=0.01,
            discrete_states_count=28
        )
        self.profile.ltp_poly_coefficients = [0.0, 0.0, 1.0, 0.0]
        self.profile.ltd_poly_coefficients = [0.0, 0.0, 1.0, 0.0]
        
    def test_vae_shapes(self):
        # Test model forward pass shapes
        model = ConvVAE(latent_dim=8, device_profile=self.profile)
        x = torch.randn(4, 1, 8, 8)
        recon_x, mu, logvar = model(x)
        
        self.assertEqual(recon_x.shape, (4, 1, 8, 8))
        self.assertEqual(mu.shape, (4, 8))
        self.assertEqual(logvar.shape, (4, 8))
        
    def test_vae_loss(self):
        # Test VAE loss calculation
        recon_x = torch.ones(2, 1, 8, 8) * 0.5
        x = torch.ones(2, 1, 8, 8)
        mu = torch.zeros(2, 8)
        logvar = torch.zeros(2, 8)
        
        loss = vae_loss_fn(recon_x, x, mu, logvar)
        self.assertTrue(loss.item() > 0)
        
    def test_self_healing_conversion(self):
        # Test converting to self-healing layers
        model = ConvVAE(latent_dim=8, device_profile=self.profile)
        convert_to_self_healing(model, self.profile)
        
        # Verify that layers are replaced
        self.assertTrue(isinstance(model.decoder_fc, SelfHealingCrossbar))
        self.assertTrue(isinstance(model.decoder_conv[1], SelfHealingConv2d))
        self.assertTrue(isinstance(model.decoder_conv[3], SelfHealingConv2d))
        
        # Test forward pass with self-healing layers
        model.eval()
        x = torch.randn(2, 1, 8, 8)
        recon_x, _, _ = model(x)
        self.assertEqual(recon_x.shape, (2, 1, 8, 8))

if __name__ == "__main__":
    unittest.main()
