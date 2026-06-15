import numpy as np

def generate_olfactory_enose_dataset(num_samples=250, seq_len=80, num_sensors=8, num_gases=4):
    """
    Generates synthetic electronic nose gas sensor array responses.
    Simulates adsorption (response rise) and desorption (response decay) phases
    with gas-specific transient kinetic parameters.
    """
    np.random.seed(42)
    t = np.linspace(0, 10, seq_len)
    t_adsorption = 4.0 # Gas exposure starts at t=0, ends at t=4.0
    
    # Define gas-specific sensitivity matrix for the 8 sensors
    # Shape: (num_gases, num_sensors)
    gas_sensitivities = np.array([
        [1.2, 0.2, 0.1, 0.8, 0.3, 0.9, 0.2, 0.5], # Gas 0 (Ethanol-like)
        [0.1, 1.5, 0.3, 0.2, 0.9, 0.1, 0.8, 0.4], # Gas 1 (Acetone-like)
        [0.4, 0.3, 1.8, 0.1, 0.2, 0.7, 0.1, 0.9], # Gas 2 (Carbon Monoxide-like)
        [0.8, 0.1, 0.2, 1.4, 0.4, 0.3, 0.9, 0.2]  # Gas 3 (Methane-like)
    ])
    
    # Gas-specific time constants for adsorption and desorption
    tau_ad = [1.5, 0.8, 2.5, 1.2]
    tau_de = [2.0, 3.5, 1.5, 4.0]
    
    X = []
    y = []
    
    for i in range(num_samples):
        gas_id = np.random.randint(0, num_gases)
        y.append(gas_id)
        
        # Sensor response array
        sensor_signals = []
        for s in range(num_sensors):
            base_sensitivity = gas_sensitivities[gas_id, s]
            ad_constant = tau_ad[gas_id] * (1.0 + np.random.normal(0, 0.1))
            de_constant = tau_de[gas_id] * (1.0 + np.random.normal(0, 0.1))
            
            signal = np.zeros(seq_len)
            for idx, time_val in enumerate(t):
                if time_val <= t_adsorption:
                    # Adsorption phase
                    signal[idx] = base_sensitivity * (1.0 - np.exp(-time_val / ad_constant))
                else:
                    # Desorption phase
                    max_ad = base_sensitivity * (1.0 - np.exp(-t_adsorption / ad_constant))
                    signal[idx] = max_ad * np.exp(-(time_val - t_adsorption) / de_constant)
                    
            # Add measurement noise
            signal += np.random.normal(0, 0.05, seq_len)
            sensor_signals.append(signal)
            
        # Shape: (seq_len, num_sensors)
        X.append(np.stack(sensor_signals, axis=1))
        
    return np.array(X), np.array(y)
