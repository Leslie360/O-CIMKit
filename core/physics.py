import torch

def inject_gaussian_noise(w_phys, noise_std):
    """
    Injects Cycle-to-Cycle (C2C) read/write noise into physical weight values.
    
    Args:
        w_phys (torch.Tensor): Weights in physical conductance/current domain.
        noise_std (float): Standard deviation of the absolute physical noise.
        
    Returns:
        torch.Tensor: Noisy physical weights.
    """
    if noise_std <= 0:
        return w_phys
    noise = torch.randn_like(w_phys) * noise_std
    return w_phys + noise

def inject_poisson_shot_noise(x_flux, max_photons=1000):
    """
    Simulates Poisson shot noise for biological vision sensors under low light.
    
    Args:
        x_flux (torch.Tensor): Input light intensity normalized to [0, 1].
        max_photons (float): Peak photon count matching brightest pixel.
        
    Returns:
        torch.Tensor: Noisy input light intensity.
    """
    # Scale to photon flux count
    photons = x_flux * max_photons
    # Sample from Poisson distribution
    noisy_photons = torch.poisson(photons)
    # Scale back to math domain
    return noisy_photons / max_photons

def apply_non_linear_gradient(grad, w_math, ltp_poly, ltd_poly, alpha=0.3, use_abs=False, normalize=False):
    """
    Modifies backpropagation gradients based on the LTP/LTD non-linear slope.
    This simulates asymmetric and non-linear conductance updates.
    
    Args:
        grad (torch.Tensor): Original mathematical gradient from autograd.
        w_math (torch.Tensor): Current weight values in mathematical domain [-1, 1].
        ltp_poly (list): Polynomial coefficients for LTP.
        ltd_poly (list): Polynomial coefficients for LTD.
        alpha (float): Strength of the physical non-linear update influence.
        use_abs (bool): Whether to take the absolute value of the slope. Default False.
        normalize (bool): Whether to normalize the slope by its maximum. Default False.
        
    Returns:
        torch.Tensor: Modified physical-aware gradient.
    """
    # Clamp weights to safe range [-1, 1] to avoid wild polynomial values
    w_safe = torch.clamp(w_math, -1.0, 1.0)
    
    # Map weights to [0, 1] for polynomial lookup
    w_norm = (w_safe + 1.0) / 2.0
    
    # Evaluate 3rd-order polynomial directly: c0*x^3 + c1*x^2 + c2*x + c3
    ltp_slope = ltp_poly[0] * (w_norm ** 3) + ltp_poly[1] * (w_norm ** 2) + ltp_poly[2] * w_norm + ltp_poly[3]
    ltd_slope = ltd_poly[0] * (w_norm ** 3) + ltd_poly[1] * (w_norm ** 2) + ltd_poly[2] * w_norm + ltd_poly[3]
    
    if use_abs:
        ltp_slope = torch.abs(ltp_slope)
        ltd_slope = torch.abs(ltd_slope)
    
    if normalize:
        # Normalize slopes to prevent gradient vanishing
        ltp_max = ltp_slope.max()
        ltd_max = ltd_slope.max()
        if ltp_max > 0:
            ltp_slope = ltp_slope / ltp_max
        if ltd_max > 0:
            ltd_slope = ltd_slope / ltd_max
        
    # Positive gradient means weight decrease (LTD is triggered)
    # Negative gradient means weight increase (LTP is triggered)
    poly_factor = torch.where(
        grad > 0,
        ltd_slope,
        ltp_slope
    )
    
    # Residual update rule: New_Grad = Original_Grad * ((1 - alpha) + alpha * poly_factor)
    modified_grad = grad * ((1.0 - alpha) + alpha * poly_factor)
    return modified_grad

def simulate_conductance_drift(w_phys, time_hours, drift_exponent=0.08, retention_noise=0.01, phys_min=0.0, phys_max=1.0):
    """
    Simulates resistance/conductance drift and noise degradation in memristive crossbars over time.
    Model: G(t) = G(t0) * (t/t0)^(-nu) + State_Noise
    
    Args:
        w_phys (torch.Tensor): Weights in physical conductance domain.
        time_hours (float): Time elapsed in hours (t0 = 1.0 hour).
        drift_exponent (float): Exponent governing drift speed (nu).
        retention_noise (float): Retention noise standard deviation.
        phys_min (float): Minimum physical conductance limit.
        phys_max (float): Maximum physical conductance limit.
    """
    if time_hours <= 1.0:
        return w_phys
        
    # Apply power-law drift
    factor = (time_hours / 1.0) ** (-drift_exponent)
    w_drift = w_phys * factor
    
    # Inject retention noise (cumulative state degradation)
    if retention_noise > 0:
        noise_std_abs = retention_noise * (phys_max - phys_min) * (time_hours ** 0.15)
        noise = torch.randn_like(w_drift) * noise_std_abs
        w_drift = w_drift + noise
        
    # Clamp to physical limits
    return torch.clamp(w_drift, min=phys_min, max=phys_max)


