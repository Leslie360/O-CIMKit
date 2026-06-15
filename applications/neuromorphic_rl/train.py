import os
import sys
import time
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from profiles.device_profile import DeviceProfile
from core.layers import QATMLPLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CartPoleSimulator:
    """
    Self-contained CartPole physics simulator using Euler-Maruyama integration.
    Avoids external gym dependencies while maintaining exact physical equations.
    """
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02 # seconds between state updates
        
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        
        self.reset()
        
    def reset(self):
        # State: [x, x_dot, theta, theta_dot]
        self.state = np.random.uniform(-0.05, 0.05, size=(4,))
        self.steps = 0
        return self.state
        
    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        # Euler integration
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1
        
        # Check termination conditions
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or self.steps >= 200
        )
        
        reward = 1.0 if not done else 0.0
        return self.state, reward, done

class QNetwork(nn.Module):
    """
    Q-Value Approximation Network with optional hardware quantization.
    """
    def __init__(self, state_dim=4, action_dim=2, device_profile=None):
        super().__init__()
        # Intermediate layers mapped to memristive crossbars with discrete quantization
        self.fc1 = QATMLPLayer(state_dim, 64, device_profile=device_profile, mode="lsq")
        self.fc2 = QATMLPLayer(64, 32, device_profile=device_profile, mode="lsq")
        self.out = nn.Linear(32, action_dim) # Software readout layer
        
    def forward(self, x):
        # x shape: (B, 4)
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return self.out(h2)

class DQNAgent:
    def __init__(self, state_dim=4, action_dim=2, device_profile=None, lr=0.002, gamma=0.99):
        self.q_net = QNetwork(state_dim, action_dim, device_profile=device_profile).to(device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.action_dim = action_dim
        
    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return q_values.argmax(dim=-1).item()
        
    def update(self, state, action, reward, next_state, done):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        
        q_values = self.q_net(state_t)
        with torch.no_grad():
            next_q_values = self.q_net(next_state_t)
            max_next_q = next_q_values.max(dim=-1)[0]
            target_q = reward + (1.0 - done) * self.gamma * max_next_q
            
        loss = nn.functional.mse_loss(q_values[0, action], target_q.squeeze())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Inject physical cycle-to-cycle weight perturbation to simulate memristor noise
        if hasattr(self.q_net.fc1, 'profile') and self.q_net.fc1.profile is not None:
            noise_std = self.q_net.fc1.profile.get_noise_std()
            if noise_std > 0:
                with torch.no_grad():
                    for name, param in self.q_net.named_parameters():
                        if "weight" in name and ("fc1" in name or "fc2" in name):
                            param.add_(torch.randn_like(param) * noise_std * 0.01)

def run_rl_training(profile, args, track_name="Ideal Float"):
    env = CartPoleSimulator()
    agent = DQNAgent(state_dim=4, action_dim=2, device_profile=profile, lr=args.lr)
    
    epsilon = 0.9
    epsilon_decay = 0.98
    epsilon_min = 0.05
    
    rewards_history = []
    
    print(f"  Training DQN controller ({track_name})...")
    for episode in range(args.episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
        rewards_history.append(episode_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        if (episode + 1) % 25 == 0:
            avg_rew = np.mean(rewards_history[-25:])
            print(f"    Episode {episode+1:03d} - Avg Reward (Last 25): {avg_rew:.1f}")
            
    # Final evaluation: run 20 test episodes
    test_rewards = []
    for _ in range(20):
        state = env.reset()
        r = 0.0
        done = False
        while not done:
            action = agent.select_action(state, epsilon=0.0) # Greedy
            state, reward, done = env.step(action)
            r += reward
        test_rewards.append(r)
        
    return np.mean(test_rewards)

def main():
    parser = argparse.ArgumentParser(description="Neuromorphic Reinforcement Learning CartPole CIM Simulation")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--epochs", type=int, default=None, help="Alias for episodes (for benchmark compatibility)")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    
    if "ipykernel" in sys.modules or not sys.argv[0].endswith("train.py"):
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
        
    if args.epochs is not None:
        args.episodes = args.epochs * 30
        
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    profile_path = os.path.join(project_root, "profiles", "repository", "FingerMemristor.json")
    profile = DeviceProfile.from_json(profile_path) if os.path.exists(profile_path) else None
    
    print("=" * 60)
    print("CIM Platform - Reinforcement Learning CartPole Controller")
    print("=" * 60)
    if profile:
        print(f"  Loaded Device Profile: {profile.device_name}")
        print(f"  discrete states count: {profile.discrete_states_count}")
    else:
        print("  ⚠️ FingerMemristor profile not found! Running standard DQN baseline.")
        
    t0 = time.time()
    # 1. Ideal float DQN
    print("📢 Phase 1: Training Ideal Software Float Baseline...")
    avg_reward_float = run_rl_training(None, args, track_name="Ideal Float")
    
    # 2. Hardware-aware DQN
    print("\n📢 Phase 2: Training Hardware-Aware DQN under Memristive Constraints...")
    avg_reward_hw = run_rl_training(profile, args, track_name="Memristive HW")
    
    duration = time.time() - t0
    
    print("\n  SIMULATION RESULTS & COMPARISON")
    print("=" * 60)
    print(f"  {'Metric':<25} | {'Ideal Float':<12} | {'Memristive HW':<16}")
    print("-" * 60)
    print(f"  {'Avg Balance Steps':<25} | {avg_reward_float:<12.1f} | {avg_reward_hw:<16.1f}")
    print(f"  {'Performance Loss (%)':<25} | {'N/A':<12} | {(1.0 - avg_reward_hw/avg_reward_float)*100.0:<16.2f}")
    print("=" * 60)
    print(f"🏆 Final DQN Balance Steps: {avg_reward_hw:.1f} / 200.0")
    print(f"⏱️ Execution time: {duration:.2f}s")
    print("=" * 60)

if __name__ == "__main__":
    main()
