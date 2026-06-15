import numpy as np

def fit_ltp_ltd(ltp_raw, ltd_raw, deg=3):
    """
    Fits 3rd-order polynomials for LTP and LTD conductance changes.
    
    Args:
        ltp_raw (list or np.ndarray): Raw conductance/current values during LTP pulses.
        ltd_raw (list or np.ndarray): Raw conductance/current values during LTD pulses.
        deg (int): Polynomial degree.
        
    Returns:
        dict: Fitted polynomial coefficients, min/max values, and fit statistics.
    """
    ltp_arr = np.abs(np.array(ltp_raw))
    ltd_arr = np.abs(np.array(ltd_raw))
    
    g_min = float(min(ltp_arr.min(), ltd_arr.min()))
    g_max = float(max(ltp_arr.max(), ltd_arr.max()))
    
    # Normalize to [0, 1]
    ltp_norm = (ltp_arr - g_min) / (g_max - g_min)
    ltd_norm = (ltd_arr - g_min) / (g_max - g_min)
    
    # Calculate step differences (delta updates)
    delta_ltp = np.diff(ltp_norm)
    delta_ltd = np.diff(ltd_norm)
    
    # Fit poly: Delta = P(Normalized_Value)
    # We predict delta based on the starting value of each transition
    ltp_poly = np.polyfit(ltp_norm[:-1], delta_ltp, deg)
    ltd_poly = np.polyfit(ltd_norm[:-1], delta_ltd, deg)
    
    return {
        "conductance_min": g_min,
        "conductance_max": g_max,
        "ltp_poly_coefficients": ltp_poly.tolist(),
        "ltd_poly_coefficients": ltd_poly.tolist()
    }

def fit_volatile_decay(time_steps, current_values):
    """
    Fits the volatile short-term decay to extract the relaxation time constant tau.
    Assume simple exponential decay model: I(t) = I_0 * exp(-t/tau) + I_offset
    
    Args:
        time_steps (np.ndarray): Time steps in seconds.
        current_values (np.ndarray): Current decay values.
        
    Returns:
        float: Fitted relaxation time constant tau (seconds).
    """
    # Simple linear regression on log(current) for simplified fitting
    # Offset subtraction can be done based on the last value
    offset = current_values[-1]
    y = current_values[:-1] - offset
    
    # Filter non-positive values to prevent log error
    valid_idx = y > 0
    if not np.any(valid_idx):
        return 1.0  # Fallback default time constant
        
    t_fit = time_steps[:-1][valid_idx]
    y_fit = np.log(y[valid_idx])
    
    # Linear fit: log(y) = -1/tau * t + log(I_0)
    slope, intercept = np.polyfit(t_fit, y_fit, 1)
    
    if slope >= 0:
        return 1.0  # Fallback if no decay is found
        
    tau = -1.0 / slope
    return float(tau)
