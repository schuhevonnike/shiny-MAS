import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.nn import ReLU
from torch.nn import Softmax
from torch import unsqueeze

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, state):
        x = torch.ReLU(self.fc1(state))
        x = torch.ReLU(self.fc2(x))
        action_prob = torch.Softmax(self.fc3(x), dim=-1)
        state_value = self.fc4(x)
        return action_prob, state_value

class PPO:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.actor_critic = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=3e-4)
        self.replay_buffer = []

        def select_action(self, state, evaluation=False):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action_prob, _ = self.actor_critic(state)
            action = np.random.choice(len(action_prob.cpu().numpy().flatten()), p=action_prob.cpu().numpy().flatten())
            return action

        def update(self, state, action, reward, next_state, done):
            self.replay_buffer.append((state, action, reward, next_state, done))
            if len(self.replay_buffer) < 1000:
                return

            state, action, reward, next_state, done = zip(*self.replay_buffer)
            self.replay_buffer = []

            state = torch.FloatTensor(state).to(device)
            action = torch.LongTensor(action).to(device)
            reward = torch.FloatTensor(reward).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(done).to(device)

            _, next_state_values = self.actor_critic(next_state)
            _, state_values = self.actor_critic(state)

            advantages = reward + (1 - done) * 0.99 * next_state_values.squeeze() - state_values.squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            action_prob, state_values = self.actor_critic(state)
            action_prob = action_prob.gather(1, action.unsqueeze(1)).squeeze(1)
            old_action_prob = action_prob.clone().detach()

            ratio = action_prob / old_action_prob
            sur1 = ratio * advantages
            sur2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages

            actor_loss = -torch.min(sur1, sur2).mean()
            critic_loss = ((reward + (1 - done) * 0.99 * next_state_values.squeeze() - state_values.squeeze()) ** 2).mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * action_prob.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
