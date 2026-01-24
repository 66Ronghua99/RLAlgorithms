import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        layers = []
        input_size = state_dim
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Return raw logits
        return self.network(x)

    def sample_action(self, state):
        with torch.no_grad():
            logits = self.forward(state)
            dist = Categorical(logits=logits)
            return dist.sample().item()

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_layers):
        super().__init__()
        layers = []
        input_size = state_dim
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Return raw value
        return self.network(x)

def compute_gae(rewards, values, dones, gamma, lam):
    # values should have length len(rewards) + 1 (bootstrapped value at end)
    advantages = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    return advantages

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # Hyperparameters
    gamma = 0.99
    lam = 0.95
    epsilon = 0.2
    lr = 0.001
    epochs = 10000
    batch_size = 500  # Steps per update
    update_epochs = 5 

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    # Record video every 500 episodes
    env = RecordVideo(env, video_folder="ppo_videos", episode_trigger=lambda x: x % 500 == 0, disable_logger=True)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = PolicyNetwork(obs_dim, [256, 256, 256], action_dim).to(device)
    critic = ValueNetwork(obs_dim, [256, 256, 256]).to(device)
    
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)

    step_bar = tqdm(range(epochs))

    for _ in step_bar:
        # --- Collection Phase ---
        states = []
        actions = []
        rewards = []
        dones = []
        
        obs, _ = env.reset()
        collected_steps = 0
        
        while collected_steps < batch_size:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            action = actor.sample_action(obs_tensor)
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(float(done))
            
            obs = next_obs
            collected_steps += 1
            
            if done:
                obs, _ = env.reset()
        
        # --- Processing Phase ---
        # Convert to tensors
        states_t = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions_t = torch.tensor(np.array(actions), dtype=torch.long).to(device)
        rewards_t = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        dones_t = torch.tensor(np.array(dones), dtype=torch.float32).to(device)
        
        # Calculate Values and GAE
        with torch.no_grad():
            values = critic(states_t).squeeze(-1)
            # Bootstrapped value for the last next_state
            next_val = critic(torch.tensor(obs, dtype=torch.float32).to(device)).item()
            values_all = torch.cat([values, torch.tensor([next_val]).to(device)])
        
        advantages = compute_gae(rewards_t, values_all, dones_t, gamma, lam)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Store old log probs for PPO ratio
        with torch.no_grad():
            logits = actor(states_t)
            dist = Categorical(logits=logits)
            old_log_probs = dist.log_prob(actions_t)

        # --- Update Phase ---
        for _ in range(update_epochs):
            # Re-evaluate
            logits = actor(states_t)
            dist = Categorical(logits=logits)
            cur_log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()
            
            cur_values = critic(states_t).squeeze(-1)
            
            # PPO Loss
            ratio = torch.exp(cur_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
            p_loss = -torch.min(surr1, surr2).mean()
            
            # Value Loss
            v_loss = F.mse_loss(cur_values, returns)
            
            loss = p_loss + 0.5 * v_loss - 0.01 * entropy
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step_bar.set_description(f"Loss: {loss.item():.4f}")

if __name__ == '__main__':
    train()
