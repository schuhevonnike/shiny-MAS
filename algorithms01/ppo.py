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
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)


class PPOAgent:
    def __init__(self, state_size, action_size, cooperative=False, learning_rate=1e-3, gamma=0.99, epsilon=0.2, update_steps=10):
        self.cooperative = cooperative  # Cooperative flag
        self.policy = PPOPolicy(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_steps = update_steps
        self.memory = []

    def act(self, state):
        """Select an action based on the current state."""
        state = state.clone().detach().float().unsqueeze(0)
        #state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def remember(self, state, action, reward, next_state, done, log_prob):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done, log_prob))

    def update(self):
        """Update the policy network using the PPO algorithm."""
        states, actions, rewards, next_states, dones, old_log_probs = zip(*self.memory)

        states = states.clone().detach().float()
        # states = torch.tensor(states, dtype=torch.float32)
        actions = actions.clone().detach().float()
        # actions = torch.tensor(actions, dtype=torch.float32)
        rewards = rewards.clone().detach().float()
        # rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = next_states.clone().detach().float()
        # next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = dones.clone().detach().float()
        # dones = torch.tensor(dones, dtype=torch.float32)
        old_log_probs = torch.stack(old_log_probs)

        # Compute returns and advantages
        returns = []
        advantages = []
        R = 0
        advantage = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            R = reward + self.gamma * R * (1 - done)
            returns.append(R)
            advantage = R - advantage
            advantages.append(advantage)

        returns = torch.tensor(returns[::-1])
        advantages = torch.tensor(advantages[::-1])

        # Cooperative behavior logic could be added here. For example:
        # - Shared memory or synchronized updates between agents.
        # - Computing returns/advantages based on joint rewards.

        for _ in range(self.update_steps):
            probs = self.policy(states)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            loss = -torch.min(surr1, surr2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory = []

# Additional methods and logic for cooperative learning could be added.
