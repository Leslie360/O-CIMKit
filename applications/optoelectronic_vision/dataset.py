import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from applications.cifar10_vision.dataset import get_dataloaders

def simulate_physical_optic(x, max_photons=10000.0):
    """
    Simulates physical optic capture in photoelectric synapses.
    Maps ideal normalized input [0, 1] to photons under ambient light intensity (max_photons),
    injects physical Poisson shot noise, and normalizes back.
    """
    # Responsivity weights for RGB bands (wavelength-selective response)
    responsivity = torch.tensor([0.706, 1.0, 0.758], device=x.device).view(1, 3, 1, 1)
    
    # Physical photon counts hitting the OECT visual pixels
    photons = x * max_photons * responsivity
    
    # Poisson noise approximated by Gaussian noise for continuous gradient flow
    noise = torch.randn_like(photons) * torch.sqrt(torch.clamp(photons, min=0.0))
    noisy_photons = photons + noise
    
    # Normalize back to mathematical range [0, 1]
    return torch.clamp(noisy_photons / max_photons, 0.0, 1.0)
