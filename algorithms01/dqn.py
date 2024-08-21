import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class QNetworkBase(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetworkBase, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAdversary(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super().__init__()
        self.q_network = QNetworkBase(state_dim, action_dim).to(device)
        self.target_q_network = QNetworkBase(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.replay_buffer = []
        self.action_dim = action_dim
        self.gamma = 0.99
        self.epsilon = 0.1
        self.device = device

    def select_action(self, state, evaluation=False):
        if not evaluation and random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_network(state).argmax().item()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) < 1000:
            return
        if len(self.replay_buffer) > 10000:
            self.replay_buffer.pop(0)

        batch = random.sample(self.replay_buffer, 64)
        state, action, reward, next_state, done = zip(*batch)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        q_values = self.q_network(state).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target_q_values = reward + (1 - done) * self.gamma * self.target_q_network(next_state).max(1)[0]
        loss = self.criterion(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if random.random() < 0.1:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

class DQNCooperator(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super().__init__()
        self.q_network = QNetworkBase(state_dim, action_dim).to(device)
        self.target_q_network = QNetworkBase(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.replay_buffer = []
        self.action_dim = action_dim
        self.gamma = 0.99
        self.epsilon = 0.1
        self.device = device

    def select_action(self, state, evaluation=False):
        if not evaluation and random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_network(state).argmax().item()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) < 1000:
            return
        if len(self.replay_buffer) > 10000:
            self.replay_buffer.pop(0)

        batch = random.sample(self.replay_buffer, 64)
        state, action, reward, next_state, done = zip(*batch)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        q_values = self.q_network(state).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target_q_values = reward + (1 - done) * self.gamma * self.target_q_network(next_state).max(1)[0]
        loss = self.criterion(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if random.random() < 0.1:
            self.target_q_network.load_state_dict(self.q_network.state_dict())