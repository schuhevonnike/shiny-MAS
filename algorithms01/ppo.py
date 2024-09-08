import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class PPOPolicy(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(PPOPolicy, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.action_head(x), dim=-1)
        state_value = self.value_head(x)
        return action_probs, state_value


class PPOAgent:
    def __init__(self, state_size, action_size, cooperative=False, learning_rate=1e-3, gamma=0.99, epsilon=0.2, update_steps=10):
        self.cooperative = cooperative
        self.policy = PPOPolicy(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_steps = update_steps
        self.memory = []

    def act(self, state):
        """Select an action based on the current state."""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probs, _ = self.policy(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, batch_size):
        """Update the policy network using the PPO algorithm."""
        states, actions, rewards, next_states, dones, old_log_probs = zip(*self.memory)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        old_log_probs = torch.stack(old_log_probs)

        # Compute returns and advantages using the value network
        _, state_values = self.policy(states)
        returns = []
        R = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            R = reward + self.gamma * R * (1 - done)
            returns.insert(0, R)

        returns = torch.tensor(returns)
        advantages = returns - state_values.detach()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.update_steps):
            action_probs, _ = self.policy(states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            ratio = (new_log_probs - old_log_probs).exp()

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            loss = -torch.min(surr1, surr2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory = []

    def remember(self, state, action, reward, next_state, done, old_log_prob):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done, old_log_prob))
