import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorNetwork, self).__init__()
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
        self.actors = [ActorNetwork(state_dim, action_dim, max_action).to(device) for _ in range(n_agents)]
        self.critics = [CriticNetwork(state_dim, action_dim, n_agents).to(device) for _ in range(n_agents)]
        self.target_actors = [ActorNetwork(state_dim, action_dim, max_action).to(device) for _ in range(n_agents)]
        self.target_critics = [CriticNetwork(state_dim, action_dim, n_agents).to(device) for _ in range(n_agents)]
        for i in range(n_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())
        self.optimizers_actor = [optim.Adam(self.actors[i].parameters(), lr=learning_rate) for i in range(n_agents)]
        self.optimizers_critic = [optim.Adam(self.critics[i].parameters(), lr=learning_rate) for i in range(n_agents)]
        self.gamma = gamma
        self.tau = tau
        self.n_agents = n_agents
        self.replay_buffer = deque(maxlen=100000)

        def select_action(self, state, agent_idx):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = self.actors[agent_idx](state)
            return action.cpu().numpy()[0]

        def store_transition(self, states, actions, rewards, next_states, dones):
            self.replay_buffer.append((states, actions, rewards, next_states, dones))

        def update(self):
            if len(self.replay_buffer) < 64:
                return

            batch = random.sample(self.replay_buffer, 64)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(states).to(device)
            actions = torch.FloatTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).to(device)

            for i in range(self.n_agents):
                with torch.no_grad():
                    next_actions = torch.cat([self.target_actors[j](next_states[:, j, :]) for j in range(self.n_agents)], dim=1)
                    target_q_values = self.target_critics[i](next_states.view(next_states.size(0), -1), next_actions)
                    y = rewards[:, i] + self.gamma * target_q_values * (1 - dones[:, i])

                q_values = self.critics[i](states.view(states.size(0), -1), actions.view(actions.size(0), -1))
                critic_loss = nn.MSELoss()(q_values, y)

                self.optimizers_critic[i].zero_grad()
                critic_loss.backward()
                self.optimizers_critic[i].step()

                actions_pred = torch.cat([self.actors[j](states[:, j, :]) if j == i else actions[:, j, :] for j in range(self.n_agents)], dim=1)
                actor_loss = -self.critics[i](states.view(states.size(0), -1), actions_pred).mean()

                self.optimizers_actor[i].zero_grad()
                actor_loss.backward()
                self.optimizers_actor[i].step()

                for target_param, param in zip(self.target_critics[i].parameters(), self.critics[i].parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for target_param, param in zip(self.target_actors[i].parameters(), self.actors[i].parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        def save(self, filename, agent_idx):
            torch.save(self.actors[agent_idx].state_dict(), filename)

        def load(self, filename, agent_idx):
            self.actors[agent_idx].load_state_dict(torch.load(filename))
