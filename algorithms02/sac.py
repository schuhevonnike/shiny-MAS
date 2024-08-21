import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class ActorAdversary(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorAdversary, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))

class ActorCooperator(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorCooperator, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class SAC:
    def __init__(self, state_dim, action_dim, max_action, device, learning_rate=0.0003, gamma=0.99, tau=0.005, alpha=0.2):
        self.device = device
        self.actor_adversary = ActorAdversary(state_dim, action_dim, max_action).to(device)
        self.actor_cooperator = ActorCooperator(state_dim, action_dim, max_action).to(device)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.value = ValueNetwork(state_dim).to(device)
        self.target_value = ValueNetwork(state_dim).to(device)
        self.target_value.load_state_dict(self.value.state_dict())
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.optimizer_actor_adversary = optim.Adam(self.actor_adversary.parameters(), lr=learning_rate)
        self.optimizer_actor_cooperator = optim.Adam(self.actor_cooperator.parameters(), lr=learning_rate)
        self.optimizer_critic_1 = optim.Adam(self.critic_1.parameters(), lr=learning_rate)
        self.optimizer_critic_2 = optim.Adam(self.critic_2.parameters(), lr=learning_rate)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=learning_rate)
        self.replay_buffer = deque(maxlen=100000)

    def select_action(self, state, agent_type, evaluation=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if agent_type == 'adversary':
            with torch.no_grad():
                action = self.actor_adversary(state)
        else:
            with torch.no_grad():
                action = self.actor_cooperator(state)
        if not evaluation:
            action += torch.normal(0, self.max_action * 0.1, size=action.size(), device=self.device)
        return action.cpu().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done, agent_type):
        self.replay_buffer.append((state, action, reward, next_state, done, agent_type))

    def update(self):
        if len(self.replay_buffer) < 64:
            return

        batch = random.sample(self.replay_buffer, 64)
        states, actions, rewards, next_states, done, agent_types = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        with torch.no_grad():
            next_actions_adversary = self.actor_adversary(next_states)
            next_actions_cooperator = self.actor_cooperator(next_states)
            target_values = self.target_value(next_states)
            target_q_values_1 = self.critic_1(next_states, torch.where(agent_types == 'adversary', next_actions_adversary, next_actions_cooperator))
            target_q_values_2 = self.critic_2(next_states, torch.where(agent_types == 'adversary', next_actions_adversary, next_actions_cooperator))
            target_q_values = torch.min(target_q_values_1, target_q_values_2) - self.alpha * target_values

        q_values_1 = self.critic_1(states, actions)
        q_values_2 = self.critic_2(states, actions)
        critic_loss_1 = nn.MSELoss()(q_values_1, rewards + self.gamma * target_q_values * (1 - done))
        critic_loss_2 = nn.MSELoss()(q_values_2, rewards + self.gamma * target_q_values * (1 - done))

        values = self.value(states)
        value_loss = nn.MSELoss()(values, target_q_values)

        policy_loss_adversary = (self.alpha * values - self.critic_1(states, self.actor_adversary(states))).mean()
        policy_loss_cooperator = (self.alpha * values - self.critic_1(states, self.actor_cooperator(states))).mean()

        self.optimizer_critic_1.zero_grad()
        critic_loss_1.backward()
        self.optimizer_critic_1.step()

        self.optimizer_critic_2.zero_grad()
        critic_loss_2.backward()
        self.optimizer_critic_2.step()

        self.optimizer_value.zero_grad()
        value_loss.backward()
        self.optimizer_value.step()

        self.optimizer_actor_adversary.zero_grad()
        policy_loss_adversary.backward()
        self.optimizer_actor_adversary.step()

        self.optimizer_actor_cooperator.zero_grad()
        policy_loss_cooperator.backward()
        self.optimizer_actor_cooperator.step()

        for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename, agent_type):
        if agent_type == 'adversary':
            torch.save(self.actor_adversary.state_dict(), filename)
        else:
            torch.save(self.actor_cooperator.state_dict(), filename)

    def load(self, filename, agent_type):
        if agent_type == 'adversary':
            self.actor_adversary.load_state_dict(torch.load(filename))
        else:
            self.actor_cooperator.load_state_dict(torch.load(filename))