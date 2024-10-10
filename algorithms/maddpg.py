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

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)  # Discrete action space

# Define critic network
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)  # Combined state and action
        self.fc2 = nn.Linear(hidden_size, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state, action):
        # Ensure that both state and action are 2D tensors with batch size as the first dimension
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        x = torch.cat([state, action], dim=1)  # Concatenate state and action
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class MADDPGAgent:
    def __init__(self, agent_id, state_size, action_size, actor_lr=1e-3, critic_lr=1e-3, gamma=0.999, tau = 1.0, tau_decay = 0.995, tau_min=0.01): # Tau for soft updating the network
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size

        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size, action_size)  # Assuming centralized critic with both agents' states/actions
        self.target_actor = Actor(state_size, action_size)
        self.target_critic = Critic(state_size, action_size)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau  # Initial soft update parameter
        self.tau_decay = tau_decay
        self.tau_min = tau_min

        self.memory = deque(maxlen=10000)

        self._hard_update(self.actor, self.target_actor)
        self._hard_update(self.critic, self.target_critic)

    def update(self, other_agent_id, agents, batch_size=32):
        # Only update if we have enough samples in memory
        if len(self.memory) < batch_size:
            return  # Not enough samples to perform an update

        # Update actor and critic networks
        experiences = random.sample(self.memory, batch_size)

        # Convert experience tuples to tensors
        states = torch.tensor(np.array([t[0] for t in experiences]), dtype=torch.float32)
        actions = torch.tensor(np.array([t[1] for t in experiences]), dtype=torch.long)
        rewards = torch.tensor(np.array([t[2] for t in experiences]), dtype=torch.float32)
        next_states = torch.tensor(np.array([t[3] for t in experiences]), dtype=torch.float32)
        dones = torch.tensor(np.array([t[4] for t in experiences]), dtype=torch.bool)

        # Gather next agents from the current agent
        next_actions = torch.cat([agents[agent].target_actor(next_states) for agent in agents if agent == self.agent_id])
        # print(f"Next actions shape after concatenating: {next_actions.shape}")

        # Collect next actions from all other agents
        other_next_actions = [agents[agent_key].target_actor(next_states) for agent_key in agents if agent_key != self.agent_id]

        # Calculate target Q-values
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)
        target_q = rewards + (1 - dones.to(torch.float32)) * self.gamma * self.target_critic(next_states, next_actions)

        # Concatenate states, actions, and other actions
        #current_state_action = torch.cat([states, actions], dim=1)

        # print(f"States shape: {states.shape}")
        # print(f"Actions shape: {actions.shape}")
        #current_q = self.critic(current_states_actions)
        current_q = self.critic(states, actions)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_q, target_q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        predicted_actions = self.actor(states)
        # print(f"Predicted actions shape: {predicted_actions.shape}")
        #actor_state_action = torch.cat([states, predicted_actions] + [agents[agent_key].actor(states) for agent_key in agents if agent_key != other_agent_id], dim=1)

        #actor_loss = -self.critic(actor_state_action).mean()
        actor_loss = -self.critic(states, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self._soft_update(self.critic, self.target_critic, self.tau)
        self._soft_update(self.actor, self.target_actor, self.tau)

    def remember(self, state, action, reward, next_state, done: bool):
        # Store experience in replay buffer
        self.memory.append((state, action, reward, next_state, done))

    @staticmethod
    def _soft_update(local_model: nn.Module, target_model: nn.Module, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def _hard_update(local_model: nn.Module, target_model: nn.Module):
        target_model.load_state_dict(local_model.state_dict())
