import os
import numpy as np
import pandas as pd

def generate_mock_data():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    devices_dir = os.path.join(project_root, "data", "devices")
    os.makedirs(devices_dir, exist_ok=True)
    
    # 1. Non-volatile Memristor mock data (LTP and LTD columns)
    n_pulses = 60
    pulse_index = np.arange(n_pulses)
    # LTP: non-linear increase with some noise
    ltp_conductance = 1e-10 + (2e-9 - 1e-10) * (1 - np.exp(-0.05 * pulse_index)) + np.random.normal(0, 2e-11, n_pulses)
    # LTD: non-linear decrease with some noise
    ltd_conductance = 1e-10 + (2e-9 - 1e-10) * np.exp(-0.06 * pulse_index) + np.random.normal(0, 2e-11, n_pulses)
    
    df_nonvolatile = pd.DataFrame({
        "LTP_Conductance": ltp_conductance,
        "LTD_Conductance": ltd_conductance
    })
    
    nonvolatile_path = os.path.join(devices_dir, "raw_memristor_conductance.csv")
    df_nonvolatile.to_csv(nonvolatile_path, index=False)
    print(f"✅ Generated mock non-volatile data: {nonvolatile_path}")
    
    # 2. Volatile Device mock data (Time and Current columns)
    time_steps = np.linspace(0, 15, 100)
    # Decay with tau = 2.5 seconds
    current = 1e-6 * np.exp(-time_steps / 2.5) + 1e-8 + np.random.normal(0, 5e-9, 100)
    
    df_volatile = pd.DataFrame({
        "Time_Seconds": time_steps,
        "Current_Amperes": current
    })
    
    volatile_path = os.path.join(devices_dir, "raw_oect_decay.csv")
    df_volatile.to_csv(volatile_path, index=False)
    print(f"✅ Generated mock volatile data: {volatile_path}")

if __name__ == "__main__":
    generate_mock_data()
