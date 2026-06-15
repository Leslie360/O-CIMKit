import os
import sys
import time
import argparse
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from profiles.device_profile import DeviceProfile
from applications.neuromorphic_pid.dataset import generate_reference_trajectory
from applications.neuromorphic_pid.model import DCMotor, MemristorPIDController, BaselinePIDController

# Custom dual logger to print to both stdout and a local log file
class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()

def calculate_settling_time(t_slice, w_slice, target, tolerance, start_time, end_time):
    """
    Calculates the settling time: the time when the output enters and remains 
    within target ± tolerance% of the steady-state target.
    """
    lower_bound = target - abs(target) * tolerance
    upper_bound = target + abs(target) * tolerance
    
    # Find the last index where the signal was outside the tolerance band
    outside_indices = np.where((w_slice < lower_bound) | (w_slice > upper_bound))[0]
    if len(outside_indices) == 0:
        return 0.0  # Already within the band
    else:
        last_outside_idx = outside_indices[-1]
        # Settling time is the duration from start to this point
        if last_outside_idx >= len(t_slice) - 1:
            # Never settled within this window
            return end_time - start_time
        settled_time = t_slice[last_outside_idx + 1]
        return settled_time - start_time

def main():
    parser = argparse.ArgumentParser(description="Neuromorphic Memristive PID DC Motor Control Simulation")
    parser.add_argument("--epochs", type=int, default=1, help="Number of simulation epochs (1 is enough for control)")
    parser.add_argument("--dt", type=float, default=0.001, help="Simulation step size (seconds)")
    parser.add_argument("--load-disturbance", type=float, default=2.0, help="Disturbance torque (Nm) added at t>=5.0s")
    
    if "ipykernel" in sys.modules or not sys.argv[0].endswith("train.py"):
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
        
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_path = os.path.join(project_root, "applications", "neuromorphic_pid", "train.log")
    
    # Redirect print to log file and stdout
    sys.stdout = DualLogger(log_path)
    
    # 1. Load Device Profile (64-state non-volatile AlOx memristor)
    profile_path = os.path.join(project_root, "profiles", "repository", "PFO_AlOx_NonVolatile.json")
    profile = DeviceProfile.from_json(profile_path) if os.path.exists(profile_path) else None
    
    print("=" * 60)
    print("CIM Platform - Neuromorphic Adaptive PID Control System")
    print("=" * 60)
    if profile:
        print(f"  Loaded Device Profile: {profile.device_name} ({profile.device_type})")
        print(f"  Discrete States: {profile.discrete_states_count}")
        print(f"  C2C noise std: {profile.get_noise_std():.4e}")
    else:
        print("  ⚠️ PFO_AlOx_NonVolatile profile not found! Running standard float model.")
        
    dt = args.dt
    total_time = 10.0
    t, r = generate_reference_trajectory(total_time=total_time, dt=dt)
    
    print(f"  Simulation parameters: dt = {dt}s, duration = {total_time}s")
    print(f"  Step 1 (0.0s - 2.5s): target = 10.0 rad/s")
    print(f"  Step 2 (2.5s - 5.0s): target = 25.0 rad/s")
    print(f"  Step 3 (5.0s - 7.5s): target = 15.0 rad/s + Load Disturbance {args.load_disturbance} Nm")
    print(f"  Step 4 (7.5s - 10.0s): target = 5.0 rad/s")
    print("-" * 60)
    
    # 2. Run Baseline Control Loop (Static PID)
    motor_base = DCMotor(dt=dt)
    controller_base = BaselinePIDController(Kp=0.5, Ki=0.2, Kd=0.01, dt=dt)
    omega_base = []
    voltage_base = []
    
    t0 = time.time()
    for k in range(len(t)):
        tk = t[k]
        rk = r[k]
        
        # Add load disturbance at t >= 5.0s
        TL = args.load_disturbance if tk >= 5.0 else 0.0
        
        ek = rk - motor_base.omega
        Vk = controller_base.compute(ek)
        omega_k = motor_base.step(Vk, T_L=TL)
        
        omega_base.append(omega_k)
        voltage_base.append(Vk)
        
    # 3. Run Neuromorphic Adaptive Control Loop (Memristor PID)
    motor_mem = DCMotor(dt=dt)
    controller_mem = MemristorPIDController(
        profile=profile,
        Kp_bounds=(0.1, 8.0),
        Ki_bounds=(0.0, 15.0),
        Kd_bounds=(0.0, 2.0),
        eta_p=0.2,
        eta_i=0.05,
        eta_d=0.005,
        dt=dt
    )
    omega_mem = []
    voltage_mem = []
    Kp_hist, Ki_hist, Kd_hist = [], [], []
    
    for k in range(len(t)):
        tk = t[k]
        rk = r[k]
        
        TL = args.load_disturbance if tk >= 5.0 else 0.0
        
        ek = rk - motor_mem.omega
        Vk = controller_mem.compute(ek, update=True)
        omega_k = motor_mem.step(Vk, T_L=TL)
        
        omega_mem.append(omega_k)
        voltage_mem.append(Vk)
        
        Kp, Ki, Kd = controller_mem.get_PID_gains()
        Kp_hist.append(Kp)
        Ki_hist.append(Ki)
        Kd_hist.append(Kd)
        
    sim_time = time.time() - t0
    
    # 4. Evaluate Performance Indicators
    omega_base = np.array(omega_base)
    omega_mem = np.array(omega_mem)
    
    # IAE
    iae_base = np.sum(np.abs(r - omega_base)) * dt
    iae_mem = np.sum(np.abs(r - omega_mem)) * dt
    # IAE reduction percentage
    iae_improvement = (iae_base - iae_mem) / iae_base * 100.0
    
    # Slice arrays for individual stage metrics
    idx1 = (t >= 0.0) & (t < 2.5)
    idx2 = (t >= 2.5) & (t < 5.0)
    idx3 = (t >= 5.0) & (t < 7.5)
    
    # Stage 1 Overshoots
    over_base1 = max(0.0, np.max(omega_base[idx1]) - 10.0) / 10.0 * 100.0
    over_mem1 = max(0.0, np.max(omega_mem[idx1]) - 10.0) / 10.0 * 100.0
    
    # Stage 2 Overshoots
    over_base2 = max(0.0, np.max(omega_base[idx2]) - 25.0) / 25.0 * 100.0
    over_mem2 = max(0.0, np.max(omega_mem[idx2]) - 25.0) / 25.0 * 100.0
    
    # Settling Times (5% tolerance band)
    settle_base1 = calculate_settling_time(t[idx1], omega_base[idx1], 10.0, 0.05, 0.0, 2.5)
    settle_mem1 = calculate_settling_time(t[idx1], omega_mem[idx1], 10.0, 0.05, 0.0, 2.5)
    
    settle_base2 = calculate_settling_time(t[idx2], omega_base[idx2], 25.0, 0.05, 2.5, 5.0)
    settle_mem2 = calculate_settling_time(t[idx2], omega_mem[idx2], 25.0, 0.05, 2.5, 5.0)
    
    # Under load disturbance (t=5.0s to 7.5s, target=15.0, with TL=2.0)
    # Re-settling time under disturbance
    settle_base3 = calculate_settling_time(t[idx3], omega_base[idx3], 15.0, 0.05, 5.0, 7.5)
    settle_mem3 = calculate_settling_time(t[idx3], omega_mem[idx3], 15.0, 0.05, 5.0, 7.5)
    
    # Max speed drop under disturbance (relative drop from 15.0)
    drop_base = max(0.0, 15.0 - np.min(omega_base[idx3]))
    drop_mem = max(0.0, 15.0 - np.min(omega_mem[idx3]))
    
    # Report results
    print("  SIMULATION RESULTS & COMPARISON")
    print("=" * 60)
    print(f"  {'Metric':<35} | {'Static PID':<10} | {'Neuromorphic PID':<16}")
    print("-" * 60)
    print(f"  {'Integral Absolute Error (IAE)':<35} | {iae_base:<10.3f} | {iae_mem:<16.3f}")
    print(f"  {'Step 1 Overshoot (%)':<35} | {over_base1:<10.2f} | {over_mem1:<16.2f}")
    print(f"  {'Step 2 Overshoot (%)':<35} | {over_base2:<10.2f} | {over_mem2:<16.2f}")
    print(f"  {'Step 1 Settling Time (s)':<35} | {settle_base1:<10.3f} | {settle_mem1:<16.3f}")
    print(f"  {'Step 2 Settling Time (s)':<35} | {settle_base2:<10.3f} | {settle_mem2:<16.3f}")
    print(f"  {'Disturbance Recovery Time (s)':<35} | {settle_base3:<10.3f} | {settle_mem3:<16.3f}")
    print(f"  {'Max Speed Drop under Load (rad/s)':<35} | {drop_base:<10.2f} | {drop_mem:<16.2f}")
    print("-" * 60)
    print(f"  PID Parameter Adaptation range for Neuromorphic Controller:")
    print(f"    Kp: {np.min(Kp_hist):.2f} -> {np.max(Kp_hist):.2f}")
    print(f"    Ki: {np.min(Ki_hist):.2f} -> {np.max(Ki_hist):.2f}")
    print(f"    Kd: {np.min(Kd_hist):.2f} -> {np.max(Kd_hist):.2f}")
    print("=" * 60)
    
    # We map the IAE reduction to a tracking accuracy score for standard benchmark logging
    tracking_accuracy = iae_improvement
    print(f"🏆 Final Neuromorphic PID Accuracy: {tracking_accuracy:.2f}%")
    print(f"⏱️ Total simulation & analysis time: {sim_time:.4f}s")
    print("=" * 60)
    sys.stdout.flush()

if __name__ == "__main__":
    main()
