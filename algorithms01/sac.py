import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
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
        x = torch.relu(self.fc1(torch.cat([state, action], 1)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class SAC:
    def __init__(self, state_dim, action_dim, max_action, device):
        self.device = device
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

        self.max_action = max_action
        self.action_dim = action_dim
        self.replay_buffer = []

        def select_action(self, state, evaluation=False):
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action = self.actor(state).cpu().data.numpy().flatten()
            if not evaluation:
                action += np.random.normal(0, self.max_action * 0.1, size=self.action_dim)
            return action.clip(-self.max_action, self.max_action)

        def update(self, state, action, reward, next_state, done):
            self.replay_buffer.append((state, action, reward, next_state, done))
            if len(self.replay_buffer) > 1000:
                self.replay_buffer.pop(0)
            if len(self.replay_buffer) < 100:
                return

            batch = random.sample(self.replay_buffer, 64)
            state, action, reward, next_state, done = zip(*batch)

            state = torch.FloatTensor(np.array(state)).to(device)
            action = torch.FloatTensor(np.array(action)).to(device)
            reward = torch.FloatTensor(np.array(reward)).to(device).reshape(-1, 1)
            next_state = torch.FloatTensor(np.array(next_state)).to(device)
            done = torch.FloatTensor(np.array(done)).to(device).reshape(-1, 1)

            with torch.no_grad():
                next_action = self.actor(next_state)
                target_q1 = self.critic1(next_state, next_action)
                target_q2 = self.critic2(next_state, next_action)
                target_q = reward + (1 - done) * 0.99 * torch.min(target_q1, target_q2)

            current_q1 = self.critic1(state, action)
            current_q2 = self.critic2(state, action)

            critic1_loss = nn.MSELoss()(current_q1, target_q)
            critic2_loss = nn.MSELoss()(current_q2, target_q)

            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()

            actor_loss = -self.critic1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
