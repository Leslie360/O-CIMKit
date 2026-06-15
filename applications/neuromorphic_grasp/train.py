import os
import sys
import time
import argparse
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from profiles.device_profile import DeviceProfile
from applications.neuromorphic_grasp.dataset import generate_grasp_scenario
from applications.neuromorphic_grasp.model import GraspEnvironment, MemristorGraspReflexController, StaticGraspController

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

def main():
    parser = argparse.ArgumentParser(description="Neuromorphic Reflex Grip Control Simulation")
    parser.add_argument("--epochs", type=int, default=1, help="Number of simulation epochs")
    parser.add_argument("--dt", type=float, default=0.001, help="Simulation step (s)")
    parser.add_argument("--eta", type=float, default=12.0, help="Tactile reflex learning rate")
    
    if "ipykernel" in sys.modules or not sys.argv[0].endswith("train.py"):
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
        
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_path = os.path.join(project_root, "applications", "neuromorphic_grasp", "train.log")
    
    sys.stdout = DualLogger(log_path)
    
    # Load Device Profile (FingerMemristor - 28-state)
    profile_path = os.path.join(project_root, "profiles", "repository", "FingerMemristor.json")
    profile = DeviceProfile.from_json(profile_path) if os.path.exists(profile_path) else None
    
    print("=" * 60)
    print("CIM Platform - Neuromorphic Tactile Reflex Grip Control")
    print("=" * 60)
    if profile:
        print(f"  Loaded Device Profile: {profile.device_name} (Memristor)")
        print(f"  Discrete States: {profile.discrete_states_count}")
    else:
        print("  ⚠️ FingerMemristor profile not found! Running standard float model.")
        
    dt = args.dt
    total_time = 10.0
    t, F_ext = generate_grasp_scenario(total_time=total_time, dt=dt)
    
    print(f"  Simulation config: dt = {dt}s, total time = {total_time}s")
    print(f"  Tugging Phase 1 (3.0s - 7.0s): Pull force = 4.5 N")
    print(f"  Tugging Phase 2 (7.0s - 10.0s): Pull force = 6.0 N")
    print("-" * 60)
    
    # 1. Run Static Baseline
    env_static = GraspEnvironment(dt=dt)
    controller_static = StaticGraspController(F_grip=10.0) # Medium static grip force
    
    v_static_hist = []
    d_static_hist = []
    F_static_grip_hist = []
    
    t0 = time.time()
    for k in range(len(t)):
        F_grip = controller_static.get_grip_force()
        v, d = env_static.step(F_grip, F_ext[k])
        
        v_static_hist.append(v)
        d_static_hist.append(d)
        F_static_grip_hist.append(F_grip)
        
    # 2. Run Neuromorphic Adaptive Reflex Control
    env_mem = GraspEnvironment(dt=dt)
    controller_mem = MemristorGraspReflexController(profile=profile, eta=args.eta, dt=dt)
    
    v_mem_hist = []
    d_mem_hist = []
    F_mem_grip_hist = []
    G_mem_hist = []
    
    for k in range(len(t)):
        F_grip = controller_mem.get_grip_force()
        v, d = env_mem.step(F_grip, F_ext[k])
        
        # Adaptive reflex update based on sensory feedback
        controller_mem.update_reflex(v)
        
        v_mem_hist.append(v)
        d_mem_hist.append(d)
        F_mem_grip_hist.append(F_grip)
        G_mem_hist.append(controller_mem.G)
        
    sim_time = time.time() - t0
    
    # 3. Analyze Slippage Metrics
    d_static_final = d_static_hist[-1]
    d_mem_final = d_mem_hist[-1]
    
    # Slippage Reduction Ratio as a proxy for "Accuracy"
    slippage_reduction = (d_static_final - d_mem_final) / d_static_final * 100.0
    
    print("  SIMULATION RESULTS & COMPARISON")
    print("=" * 60)
    print(f"  {'Metric':<30} | {'Static Grip':<12} | {'Neuromorphic Reflex':<20}")
    print("-" * 60)
    print(f"  {'Final Slip Distance (cm)':<30} | {d_static_final:<12.4f} | {d_mem_final:<20.4f}")
    print(f"  {'Max Slip Velocity (cm/s)':<30} | {np.max(v_static_hist):<12.4f} | {np.max(v_mem_hist):<20.4f}")
    print(f"  {'Final Grip Force (N)':<30} | {F_static_grip_hist[-1]:<12.2f} | {F_mem_grip_hist[-1]:<20.2f}")
    print("-" * 60)
    print(f"  Memristor conductance state evolution:")
    print(f"    G_mem range: {np.min(G_mem_hist):.3e} S -> {np.max(G_mem_hist):.3e} S")
    print("=" * 60)
    
    print(f"🏆 Final Neuromorphic Grasp Accuracy: {slippage_reduction:.2f}%")
    print(f"⏱️ Total simulation & analysis time: {sim_time:.4f}s")
    print("=" * 60)
    sys.stdout.flush()

if __name__ == "__main__":
    main()
