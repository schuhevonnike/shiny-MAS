import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

# Define actor network
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Assuming continuous action space, which is true for simple_tag

# Define critic network
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
    def __init__(self, state_size, action_size, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=0.01):
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
        #self._update_target_networks(tau=1.0)

        self._hard_update(self.actor, self.target_actor)
        self._hard_update(self.critic, self.target_critic)

    @staticmethod
    def _soft_update(local_model: nn.Module, target_model: nn.Module, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def _hard_update(local_model: nn.Module, target_model: nn.Module):
        target_model.load_state_dict(local_model.state_dict())

    #def act(self, state: np.ndarray, noise: float=0.1):
    #    Select an action based on the current policy
    #    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    #    with torch.no_grad():
    #       action = self.actor(state).detach().numpy()[0]
    #       action = self.actor(state).squeeze(0).numpy()
    #    self.actor.train()
    #    action += noise * np.random.randn(*action.shape)  # Add exploration noise
    #    return np.clip(action, -1, 1)  # Assuming action space [-1, 1]

    def update(self, other_agent, batch_size=32):
        # Update actor and critic networks
        experiences = random.sample(self.memory, batch_size)
        #states, actions, rewards, next_states, dones = experiences

        # Convert experience tuples to tensors
        states = torch.FloatTensor(np.array([e[0] for e in experiences]))
        actions = torch.FloatTensor(np.array([e[1] for e in experiences]))
        rewards = torch.FloatTensor(np.array([e[2] for e in experiences]).reshape(-1, 1))
        next_states = torch.FloatTensor(np.array([e[3] for e in experiences]))
        dones = torch.FloatTensor(np.array([e[4] for e in experiences]).reshape(-1, 1))

        # Update critic
        next_actions = self.target_actor(next_states)
        other_next_actions = other_agent.target_actor(next_states)
        next_state_action = torch.cat([next_states, next_actions, other_next_actions], dim=1)
        target_q = rewards + (1 - dones) * self.gamma * self.target_critic(next_state_action)

        current_state_action = torch.cat([states, actions, other_agent.actor(states)], dim=1)
        current_q = self.critic(current_state_action)

        critic_loss = nn.MSELoss()(current_q, target_q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        predicted_actions = self.actor(states)
        actor_state_action = torch.cat([states, predicted_actions, other_agent.actor(states)], dim=1)
        actor_loss = -self.critic(actor_state_action).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self._soft_update(self.critic, self.target_critic, self.tau)
        self._soft_update(self.actor, self.target_actor, self.tau)

    def remember(self, state, action, reward, next_state, done: bool):
        # Store experience in replay buffer
        self.memory.append((state, action, reward, next_state, done))

    #def sample_memory(self, batch_size):
        # Sample a batch of experiences from memory
        #experiences = zip(*[self.memory[i] for i in np.random.choice(len(self.memory), batch_size)])
        #return experiences
    #    return random.sample(self.memory, batch_size)