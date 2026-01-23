import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np
from gymnasium.wrappers import RecordVideo
from tqdm import tqdm

class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim, hidden_layers, action_dim) -> None:
        super().__init__()
        layers = []
        
        # Input layer
        current_input_size = state_dim
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            current_input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(current_input_size, action_dim))
        
        # Use nn.Sequential to build the network
        self.network = nn.Sequential(*layers)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        output = self.network(x)
        return self.softmax(output)

    def sample_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32)
        action_log_prob = self.forward(state)
        return Categorical(logits=action_log_prob).sample().item()

def log_prob(policy, actions, states):
    states = torch.as_tensor(states, dtype=torch.float32)
    action_log_prob = policy(states)
    return action_log_prob.gather(1, actions.view(-1, 1)).squeeze(1)

def rewards_to_go(rewards):
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    rewards = rewards.flip(1)
    for i in range(1, len(rewards)):
        rewards[i] += rewards[i-1]
    rewards = rewards.flip(0)
    return rewards 

def compute_loss(log_prob, rewards):
    return - (log_prob * rewards).mean()

def train():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    lr = 0.001
    epoch = 10000
    batch_size = 500  # 建议按总步数采样，或者固定回合数
    
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    # 每 500 个 Episode 录制一次视频
    env = RecordVideo(env, video_folder="videos", episode_trigger=lambda x: x % 100 == 0, disable_logger=True)
    
    action_space = env.action_space.n
    obs_space = env.observation_space.shape[0]
    
    policy = PolicyNetwork(obs_space, [256, 256, 256], action_space)
    policy.to(device)
    policy_optim = torch.optim.Adam(policy.parameters(), lr=lr)

    # 使用 tqdm 进度条，并设置 postfix 来显示 reward
    pbar = tqdm(range(epoch))
    
    for e in pbar:
        batch_obs = []
        batch_acts = []
        batch_weights = [] # 这里的权重通常是当前步到回合结束的奖励总和
        batch_rets = []    # 存放每个 episode 的总分，用于打印观测
        
        obs, _ = env.reset()
        done = False
        ep_rewards = [] # 记录当前回合的每一帧奖励

        # 收集一个 Batch 的数据
        while True:
            batch_obs.append(obs)
            
            # 推断动作
            obs_t = torch.as_tensor(obs, dtype=torch.float32).to(device)
            action = policy.sample_action(obs_t)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            batch_acts.append(action)
            ep_rewards.append(reward)

            if done:
                # 重点：计算这个回合的总分并记录
                ep_ret = sum(ep_rewards)
                batch_rets.append(ep_ret)
                
                # 重点：计算每个动作对应的权重 (Reward-to-go)
                # 简单版本：这个动作之后获得的总奖励。CartPole 里通常直接用总分填充。
                batch_weights += [ep_ret] * len(ep_rewards)
                
                # 重置环境
                obs, _ = env.reset()
                ep_rewards = []
                
                # 当收集到足够的数据（比如 10 个 episode）就跳出循环进行更新
                if len(batch_rets) >= 5: 
                    break

        # 转换为 Tensor
        obs_tensor = torch.as_tensor(np.array(batch_obs), dtype=torch.float32).to(device)
        acts_tensor = torch.as_tensor(np.array(batch_acts), dtype=torch.int64).to(device)
        weights_tensor = torch.as_tensor(np.array(batch_weights), dtype=torch.float32).to(device)

        # 策略梯度更新
        probs = log_prob(policy, acts_tensor, obs_tensor)
        # 这里的 weights_tensor 决定了我们要“强化”哪些动作
        loss = -(probs * weights_tensor).mean() 
        
        policy_optim.zero_grad()
        loss.backward()
        policy_optim.step()

        # --- 打印与监控 ---
        avg_ret = np.mean(batch_rets)
        pbar.set_postfix({'AvgReward': f'{avg_ret:.2f}', 'Loss': f'{loss.item():.4f}'})
        
        if e % 100 == 0:
            print(f"Epoch {e}: Average Reward = {avg_ret}")
        
    
            
                
            




if __name__ == "__main__":
    train()
