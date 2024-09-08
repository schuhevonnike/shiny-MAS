import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Assuming continuous action space


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)  # Combined state and action
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # Concatenate state and action
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class MADDPGAgent:
    def __init__(self, state_size, action_size, cooperative=False, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=0.01):
        self.cooperative = cooperative
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size * 2, action_size * 2)  # Assuming centralized critic with both agents' states/actions
        self.target_actor = Actor(state_size, action_size)
        self.target_critic = Critic(state_size * 2, action_size * 2)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau  # Soft update parameter
        self.memory = deque(maxlen=10000)

        # Initialize target networks to match initial actor/critic networks
        self._update_target_networks(tau=1.0)

    def _update_target_networks(self, tau=None):
        """Soft update target networks."""
        tau = self.tau if tau is None else tau
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def act(self, state, noise=0.1):
        """Select an action based on the current policy."""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        action += noise * np.random.randn(*action.shape)  # Add exploration noise
        return np.clip(action, -1, 1)  # Assuming action space [-1, 1]

    def update(self, experiences, other_agent):
        """Update actor and critic networks."""
        states, actions, rewards, next_states, dones = experiences

        # Convert experience tuples to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Actor update (policy gradient step)
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Critic update (Q-learning step)
        with torch.no_grad():
            target_actions = self.target_actor(next_states)
            next_Q_values = self.target_critic(next_states, target_actions)
            target_Q_values = rewards + self.gamma * next_Q_values * (1 - dones)

        Q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(Q_values, target_Q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Soft update of target networks
        self._update_target_networks()

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def sample_memory(self, batch_size):
        """Sample a batch of experiences from memory."""
        experiences = zip(*[self.memory[i] for i in np.random.choice(len(self.memory), batch_size)])
        return experiences
