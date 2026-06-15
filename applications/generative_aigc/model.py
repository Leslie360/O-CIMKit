import torch
import torch.nn as nn
from core.layers import QATMLPLayer, OrganicSynapseConv

class ConvVAE(nn.Module):
    """
    A Convolutional Variational Autoencoder (VAE) for 8x8 digit generation,
    using hardware-aware layers in the decoder to simulate AIGC on CIM.
    """
    def __init__(self, latent_dim=8, device_profile=None):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder: Conv layers (Software float only for standard feature extraction)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),  # -> 8x8x8
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), # -> 16x4x4
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(16 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(16 * 4 * 4, latent_dim)
        
        # Decoder: Fully connected + Transpose Convolutions (Mapped to hardware-aware layers)
        self.decoder_fc = QATMLPLayer(latent_dim, 16 * 4 * 4, device_profile=device_profile)
        self.decoder_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'), # -> 8x8
            OrganicSynapseConv(16, 8, kernel_size=3, stride=1, padding=1, device_profile=device_profile),
            nn.ReLU(),
            OrganicSynapseConv(8, 1, kernel_size=3, stride=1, padding=1, device_profile=device_profile),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(h.size(0), 16, 4, 4)
        return self.decoder_conv(h)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
