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
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.log_std = nn.Parameter(torch.zeros(action_size))

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.fc3(x)
        std = torch.exp(self.log_std)
        return mean, std


class SACAgent:
    def __init__(self, state_size, action_size, cooperative=False, learning_rate=1e-3, gamma=0.99, alpha=0.2):
        self.cooperative = cooperative  # Cooperative flag
        self.policy = PolicyNetwork(state_size, action_size)
        self.q1 = QNetwork(state_size, action_size)
        self.q2 = QNetwork(state_size, action_size)
        self.target_q1 = QNetwork(state_size, action_size)
        self.target_q2 = QNetwork(state_size, action_size)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=learning_rate)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.alpha = alpha

        # Initialize target networks
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.memory = deque(maxlen=2000)

    def act(self, state):
        """Select an action based on the current state."""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        mean, std = self.policy(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action.squeeze(0).numpy(), dist.log_prob(action).sum(dim=-1)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def update(self, batch_size=64):
        """Update the policy and value networks using Soft Actor-Critic (SAC) algorithm."""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

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

        with torch.no_grad():
            next_actions, next_log_probs = self.act(next_states)
            target_q1 = self.target_q1(next_states, next_actions)
            target_q2 = self.target_q2(next_states, next_actions)
            target_value = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target = rewards + (1 - dones) * self.gamma * target_value

        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)
        q1_loss = nn.MSELoss()(q1, target)
        q2_loss = nn.MSELoss()(q2, target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        actions, log_probs = self.act(states)
        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)
        q = torch.min(q1, q2)
        policy_loss = (self.alpha * log_probs - q).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update target networks
        for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
            target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

        for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

        # Cooperative behavior logic could be added here. For example:
        # - Shared memory or synchronized updates between agents.
        # - Computing value updates based on joint state-action pairs.

# Additional cooperative methods and logic could be added as needed.
