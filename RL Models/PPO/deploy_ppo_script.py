import torch
import torch.nn as nn
import numpy as np
import gymnasium
from torch.serialization import add_safe_globals
add_safe_globals([gymnasium.spaces.box.Box])

# Insert the correct path to the .pth model file on the deployment system
MODEL_PATH = "ppo_frequency_model.pth"

FREQUENCIES = {
    0: 3.32,
    1: 3.34,
    2: 3.90
}

class SB3_MLP_Extractor(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64):
        super().__init__()

        # Actor (policy network)
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh()
        )

        # Critic (value network)
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh()
        )

    def forward(self, x):
        return self.policy_net(x), self.value_net(x)


class PPOPolicy(nn.Module):
    def __init__(self, obs_dim=8, act_dim=3):
        super().__init__()

        self.mlp_extractor = SB3_MLP_Extractor(obs_dim, 64)

        self.action_net = nn.Linear(64, act_dim)
        self.value_net = nn.Linear(64, 1)

    def forward(self, x):
        policy_latent, _ = self.mlp_extractor(x)
        return self.action_net(policy_latent)



def load_ppo_policy(model_path=MODEL_PATH):
    policy = PPOPolicy()
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()

    print("✓ PPO Policy Loaded")
    return policy


policy = load_ppo_policy()

# FREQUENCY PREDICTION FUNCTION
def predict_frequency(telemetry):
    
    state = np.array([
        telemetry.get("snr1", 0),
        telemetry.get("snr2", 0),
        telemetry.get("snr3", 0),
        telemetry.get("snr4", 0),
        telemetry.get("distance", 0),
        telemetry.get("lat", 0),
        telemetry.get("lon", 0),
        telemetry.get("alt", 0),
    ], dtype=np.float32)

    state = np.nan_to_num(state, nan=0.0, posinf=1000, neginf=-1000)

    x = torch.tensor(state).unsqueeze(0)
    logits = policy(x)
    action = torch.argmax(logits, dim=1).item()

    return FREQUENCIES[action]
