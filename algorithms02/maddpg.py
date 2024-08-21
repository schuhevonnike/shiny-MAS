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

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, n_agents):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim * n_agents + action_dim * n_agents, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class MADDPG:
    def __init__(self, state_dim, action_dim, n_agents, max_action, device, learning_rate=0.0003, gamma=0.99, tau=0.005):
        self.device = device
        self.actor_adversary = [ActorAdversary(state_dim, action_dim, max_action).to(device) for _ in range(n_agents)]
        self.actor_cooperator = [ActorCooperator(state_dim, action_dim, max_action).to(device) for _ in range(n_agents)]
        self.critics = [CriticNetwork(state_dim, action_dim, n_agents).to(device) for _ in range(n_agents)]
        self.target_actor_adversary = [ActorAdversary(state_dim, action_dim, max_action).to(device) for _ in range(n_agents)]
        self.target_actor_cooperator = [ActorCooperator(state_dim, action_dim, max_action).to(device) for _ in range(n_agents)]
        self.target_critics = [CriticNetwork(state_dim, action_dim, n_agents).to(device) for _ in range(n_agents)]
        for i in range(n_agents):
            self.target_actor_adversary[i].load_state_dict(self.actor_adversary[i].state_dict())
            self.target_actor_cooperator[i].load_state_dict(self.actor_cooperator[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())
        self.optimizers_actor_adversary = [optim.Adam(self.actor_adversary[i].parameters(), lr=learning_rate) for i in range(n_agents)]
        self.optimizers_actor_cooperator = [optim.Adam(self.actor_cooperator[i].parameters(), lr=learning_rate) for i in range(n_agents)]
        self.optimizers_critic = [optim.Adam(self.critics[i].parameters(), lr=learning_rate) for i in range(n_agents)]
        self.gamma = gamma
        self.tau = tau
        self.n_agents = n_agents
        self.replay_buffer = deque(maxlen=100000)

    def select_action(self, state, agent_idx, agent_type):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if agent_type == 'adversary':
            with torch.no_grad():
                action = self.actor_adversary[agent_idx](state)
        else:
            with torch.no_grad():
                action = self.actor_cooperator[agent_idx](state)
        return action.cpu().numpy()[0]

    def store_transition(self, states, actions, rewards, next_states, done):
        self.replay_buffer.append((states, actions, rewards, next_states, done))

    def update(self):
        if len(self.replay_buffer) < 64:
            return

        batch = random.sample(self.replay_buffer, 64)
        states, actions, rewards, next_states, done = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        for i in range(self.n_agents):
            with torch.no_grad():
                next_actions_adversary = torch.cat([self.target_actor_adversary[j](next_states[:, j, :]) for j in range(self.n_agents)], dim=1)
                next_actions_cooperator = torch.cat([self.target_actor_cooperator[j](next_states[:, j, :]) for j in range(self.n_agents)], dim=1)
                target_q_values = self.target_critics[i](next_states.view(next_states.size(0), -1), torch.where(actions[:, i, :] == next_actions_adversary[:, i], next_actions_adversary, next_actions_cooperator))
                y = rewards[:, i] + self.gamma * target_q_values * (1 - done[:, i])

            q_values = self.critics[i](states.view(states.size(0), -1), actions.view(actions.size(0), -1))
            critic_loss = nn.MSELoss()(q_values, y)

            self.optimizers_critic[i].zero_grad()
            critic_loss.backward()
            self.optimizers_critic[i].step()

            actions_pred = torch.cat([self.actor_adversary[j](states[:, j, :]) if j == i else actions[:, j, :] for j in range(self.n_agents)], dim=1)
            actor_loss = -self.critics[i](states.view(states.size(0), -1), actions_pred).mean()

            self.optimizers_actor_adversary[i].zero_grad()
            actor_loss.backward()
            self.optimizers_actor_adversary[i].step()

            for target_param, param in zip(self.target_critics[i].parameters(), self.critics[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for target_param, param in zip(self.target_actor_adversary[i].parameters(), self.actor_adversary[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename, agent_idx, agent_type):
        if agent_type == 'adversary':
            torch.save(self.actor_adversary[agent_idx].state_dict(), filename)
        else:
            torch.save(self.actor_cooperator[agent_idx].state_dict(), filename)

    def load(self, filename, agent_idx, agent_type):
        if agent_type == 'adversary':
            self.actor_adversary[agent_idx].load_state_dict(torch.load(filename))
        else:
            self.actor_cooperator[agent_idx].load_state_dict(torch.load(filename))