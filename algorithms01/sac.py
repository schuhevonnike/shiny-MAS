import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # Concatenate state and action
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.log_std = nn.Parameter(torch.zeros(action_size))  # Log standard deviation for action distribution

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.fc3(x)
        std = torch.exp(self.log_std)
        return mean, std


class SACAgent:
    def __init__(self, state_size, action_size, cooperative=False, learning_rate=1e-3, gamma=0.99, alpha=0.2):
        self.cooperative = cooperative  # Cooperative flag to distinguish modes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Networks for policy and Q-values
        self.policy = PolicyNetwork(state_size, action_size).to(self.device)
        self.q1 = QNetwork(state_size, action_size).to(self.device)
        self.q2 = QNetwork(state_size, action_size).to(self.device)
        self.target_q1 = QNetwork(state_size, action_size).to(self.device)
        self.target_q2 = QNetwork(state_size, action_size).to(self.device)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=learning_rate)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=learning_rate)

        self.gamma = gamma
        self.alpha = alpha

        # Initialize target networks with the same weights as Q-networks
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.memory = deque(maxlen=2000)  # Experience replay buffer

    def act(self, state):
        """Select an action based on the current state."""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        mean, std = self.policy(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action.squeeze(0).detach().cpu().numpy(), dist.log_prob(action).sum(dim=-1).detach()

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def update(self, batch_size=64, other_agents=None):
        """Update the policy and value networks using Soft Actor-Critic (SAC) algorithm."""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        #states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert experience tuples to tensors
        states = torch.tensor(np.array([t[0] for t in minibatch]), dtype=torch.float32)
        actions = torch.tensor(np.array([t[1] for t in minibatch]), dtype=torch.long)
        rewards = torch.tensor(np.array([t[2] for t in minibatch]), dtype=torch.float32)
        next_states = torch.tensor(np.array([t[3] for t in minibatch]), dtype=torch.float32)
        dones = torch.tensor(np.array([t[4] for t in minibatch]), dtype=torch.bool)

        #states = torch.tensor(states, dtype=torch.float32).to(self.device)
        #actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        #rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        #next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        #dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Compute target Q values
        with torch.no_grad():
            next_actions, next_log_probs = self.act(next_states)
            next_actions = torch.tensor(next_actions, dtype=torch.float32).to(self.device)
            target_q1 = self.target_q1(next_states, next_actions)
            target_q2 = self.target_q2(next_states, next_actions)
            target_value = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target = rewards + (1 - dones) * self.gamma * target_value

        # Update Q1
        q1 = self.q1(states, actions)
        q1_loss = nn.MSELoss()(q1, target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        # Update Q2
        q2 = self.q2(states, actions)
        q2_loss = nn.MSELoss()(q2, target)

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Update policy
        actions, log_probs = self.act(states)
        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)
        q = torch.min(q1, q2)
        policy_loss = (self.alpha * log_probs - q).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft update target networks
        self.soft_update(self.target_q1, self.q1, tau=0.005)
        self.soft_update(self.target_q2, self.q2, tau=0.005)

    def soft_update(self, target_network, source_network, tau=0.005):
        """Soft update of target network parameters."""
        for target_param, param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def load_weights(self, filepath):
        """Load model weights."""
        self.policy.load_state_dict(torch.load(filepath))

    def save_weights(self, filepath):
        """Save model weights."""
        torch.save(self.policy.state_dict(), filepath)
