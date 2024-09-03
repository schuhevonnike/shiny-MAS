import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class MADDPGAgent:
    def __init__(self, state_size, action_size, cooperative=False, learning_rate=1e-3, gamma=0.99, tau=1e-3):
        self.cooperative = cooperative  # Cooperative flag
        self.actor = Actor(state_size, action_size)
        self.target_actor = Actor(state_size, action_size)
        self.critic = Critic(state_size, action_size)
        self.target_critic = Critic(state_size, action_size)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.tau = tau

        self.memory = deque(maxlen=2000)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def act(self, state):
        """ Select action based on the state. """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = self.actor(state)
        return action.squeeze(0).detach().numpy()

    def remember(self, state, action, reward, next_state, done):
        """ Store experience in replay memory. """
        self.memory.append((state, action, reward, next_state, done))

    def update(self, batch_size=64):
        """ Update the actor and critic networks using sampled batch of experiences. """
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = states.clone().detach().float()
        #states = torch.tensor(states, dtype=torch.float32)
        actions = actions.clone().detach().float()
        #actions = torch.tensor(actions, dtype=torch.float32)
        rewards = rewards.clone().detach().float()
        #rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = next_states.clone().detach().float()
        #next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = dones.clone().detach().float()
        #dones = torch.tensor(dones, dtype=torch.float32)

        # Next action and Q-value for next states
        next_actions = self.target_actor(next_states)
        target_q = self.target_critic(next_states, next_actions)
        target = rewards + (1 - dones) * self.gamma * target_q.detach()

        # Current Q-value
        q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q_values, target)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Optimize actor
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)

    def soft_update(self, target_model, model):
        """ Soft update model parameters. """
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# Additional cooperative behavior could be added here, such as sharing observations, actions, or rewards between agents.
