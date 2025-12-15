"""
PPO Training Script 
"""

import pandas as pd
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces


# CONFIGURATION
state_features = [
    "snr_bs1", "snr_bs2", "snr_bs3", "snr_bs4",
    "distance_to_selected_bs_m",
    "lat", "lon", "alt"
]

freq_to_action = {
    3320000000.0: 0,
    3340000000.0: 1,
    3900000000.0: 2
}

action_to_freq = {0: 3.32, 1: 3.34, 2: 3.9}

#load data
df = pd.read_csv("C:/Users/Preya/Desktop/Mizzou/Sem 1/Cloud/fieldvision_rl/final_rl_logs.csv")
df = df.fillna(0)
df = df.replace([np.inf, -np.inf], 0)

df["action"] = df["channel"].map(freq_to_action).fillna(0).astype(int)
df["reward"] = (df["throughput_mbps"] - 
                0.5 * df["latency_ms"] - 
                0.5 * df["rtt_ms"] - 
                0.25 * df["delay_ms"])

print(f"Loaded {len(df)} samples")

# Performance lookup

print("Creating performance lookup...")

df["location_bin"] = pd.cut(df["lat"], bins=5, labels=False)
df["distance_bin"] = pd.cut(df["distance_to_selected_bs_m"], bins=5, labels=False)
df["snr_avg"] = (df["snr_bs1"] + df["snr_bs2"] + df["snr_bs3"] + df["snr_bs4"]) / 4
df["snr_bin"] = pd.cut(df["snr_avg"], bins=5, labels=False)

performance_by_condition = {}
for _, row in df.iterrows():
    condition = (int(row["location_bin"]), int(row["distance_bin"]), int(row["snr_bin"]))
    action = int(row["action"])
    reward = row["reward"]
    
    if condition not in performance_by_condition:
        performance_by_condition[condition] = {0: [], 1: [], 2: []}
    performance_by_condition[condition][action].append(reward)

avg_performance = {}
for condition, actions in performance_by_condition.items():
    avg_performance[condition] = {}
    for action in [0, 1, 2]:
        if len(actions[action]) > 0:
            avg_performance[condition][action] = np.mean(actions[action])
        else:
            avg_performance[condition][action] = None

#define environmnet

class FreqEnv(gym.Env):
    def __init__(self, df, avg_performance):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.avg_performance = avg_performance
        self.idx = 0
        self.max_idx = len(df) - 1
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
    
    def _get_state(self):
        idx = min(self.idx, self.max_idx)
        row = self.df.iloc[idx]
        state = np.array([float(row[f]) for f in state_features], dtype=np.float32)
        return np.nan_to_num(state, nan=0.0, posinf=1000.0, neginf=-1000.0)
    
    def _get_condition(self, row):
        return (int(row["location_bin"]), int(row["distance_bin"]), int(row["snr_bin"]))
    
    def _estimate_reward(self, action, condition, actual_reward):
        if condition in self.avg_performance:
            est_reward = self.avg_performance[condition].get(action)
            if est_reward is not None:
                return est_reward
        return actual_reward + np.random.normal(0, 5)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        return self._get_state(), {}
    
    def step(self, action):
        if self.idx >= self.max_idx:
            return self._get_state(), 0.0, True, False, {}
        
        row = self.df.iloc[self.idx]
        condition = self._get_condition(row)
        actual_action = int(row["action"])
        actual_reward = row["reward"]
        
        if action == actual_action:
            reward = actual_reward
        else:
            reward = self._estimate_reward(action, condition, actual_reward)
        
        self.idx += 1
        done = self.idx >= len(self.df)
        return self._get_state(), float(reward), done, False, {}

# TRAIN PPO MODEL
print("\nTraining PPO model...")
env = DummyVecEnv([lambda: FreqEnv(df, avg_performance)])

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=256,
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    ent_coef=0.1,
)

model.learn(total_timesteps=100000)


print("\nSaving model...")

# Extract the policy network
policy = model.policy

torch.save({
    'policy_state_dict': policy.state_dict(),
    'observation_space': env.observation_space,
    'action_space': env.action_space,
}, 'ppo_frequency_model.pth')

print("✓ Model saved as 'ppo_frequency_model.pth'")
print("\nModel training complete!")

print("Use 'deploy_ppo.py' to load and use this model on AERPAW")
