import torch
from torch.distributions import Categorical
from torch import nn
from base_rl import Policy
import numpy as np
import gym

class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size) -> None:
        super().__init__()
        layers = []
        
        # Input layer
        current_input_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            current_input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(current_input_size, output_size))
        
        # Use nn.Sequential to build the network
        self.network = nn.Sequential(*layers)


    def forward(self, x):
        output = self.network(x)
        return output


    def to(self, device=None):
        if device:
            self.device = device
        super().to(self.device)


class DiscretePolicyGradient(Policy):

    def __init__(self, input_space, policy, state_value) -> None:
        super().__init__(input_space, policy)
        self.policy_network = policy
        self.value_network = state_value

    def get_policy(self, state):
        action_vector = self.policy_network(state)
        log_softmax = torch.log_softmax(action_vector, 0)
        return log_softmax

    def get_state_value(self, state):
        return self.value_network(state)

    
def get_action(log_prob):
    return Categorical(logits=log_prob).sample().item()

def compute_loss(log_prob, rewards, values):
    return - ((log_prob) * (rewards - values)).mean()


def main(policy_network=None, value_network=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 0.001
    epoch = 10000
    from tqdm import tqdm
    env = gym.make("CartPole-v1")
    action_space = env.action_space.n
    obs_space = env.observation_space.shape[0]
    if not policy_network:
        policy_network = MLP(obs_space, [256, 256, 256], action_space)
    policy_network.to(device)
    if not value_network:
        value_network = MLP(obs_space, [256, 256, 256], 1)
    value_network.to(device)
    policy = DiscretePolicyGradient(obs_space, policy_network, value_network)
    policy_optim = torch.optim.Adam(policy_network.parameters(), lr=lr)
    value_criterion = nn.MSELoss()
    value_optim = torch.optim.Adam(value_network.parameters(), lr=lr)
    for e in tqdm(range(epoch)):
        obs, _  = env.reset()
        batch_rewards = np.array([])
        rewards = np.array([])
        observes = []
        actions = []
        iter_step = 0
        while True:
            observes.append(obs)
            obsev = torch.as_tensor(obs, device=device)
            action_log_prob = policy.get_policy(obsev)
            action = get_action(action_log_prob)
            actions.append(action)
            obs, reward, done, _, _ = env.step(action)
            if len(rewards) > 0:
                rewards += reward
            rewards = np.append(rewards, reward)
            iter_step += 1
            if done or iter_step > 1000:
                obs, _ = env.reset()
                batch_rewards = np.append(batch_rewards, rewards)
                rewards = np.array([])
                if iter_step > 100:
                    break
        actions = torch.as_tensor(actions, device=device, dtype=torch.int64).reshape(-1, 1)
        observes = torch.as_tensor(observes, device=device, dtype=torch.float32).reshape(-1,obs_space)
        log_probs = policy.get_policy(observes)
        batch_rewards = torch.as_tensor(batch_rewards, device=device, dtype=torch.float32).reshape(-1,1)
        values = policy.get_state_value(observes)

        loss = compute_loss(log_probs.gather(1, actions), batch_rewards.detach(), values.detach())
        policy_optim.zero_grad()
        loss.backward()
        policy_optim.step()

        value_loss = value_criterion(batch_rewards, values)
        value_optim.zero_grad()
        value_loss.backward()
        value_optim.step()

        if (e+1)%100 == 0:
            print("value loss:", value_loss.item())
            obs, _  = env.reset()
            actions = []
            step_num = 0
            while True:
                obsev = torch.as_tensor(obs, device=device)
                action_log_prob = policy.get_policy(obsev)
                action = get_action(action_log_prob)
                actions.append(action)
                obs, reward, done, _, _ = env.step(action)
                step_num += 1
                if done or step_num > 10000:
                    obs, _ = env.reset()
                    print(f"Last for {len(actions)} actions")
                    break
            if step_num > 10000:
                break
    
    torch.save(policy_network, "cartpole-PG-policy")
    torch.save(value_network, "cartpole-PG-value")
        
            
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-p', '--policy', type=str, required=False, help='')
parser.add_argument('-v', '--value', type=str, required=False, help='')
args = parser.parse_args()


policy_network = None
value_network = None
if args.policy:
    policy_network = torch.load("cartpole-PG-policy")
if args.value:
    value_network = torch.load("cartpole-PG-value")

main(policy_network=policy_network, value_network=value_network)

        
        
            


