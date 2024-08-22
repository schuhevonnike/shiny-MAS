import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from pettingzoo.utils import agent_selector
from rlcard.games.doudizhu.utils import action


class QNetworkBase(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetworkBase, self).__init__()
        # Initialize input layers
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
        # Initialize adversary actors
        self.actor = QNetworkBase(state_dim, action_dim).to(device)
        self.target_q_network = QNetworkBase(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.replay_buffer = []
        self.action_dim = action_dim
        self.gamma = 0.99
        self.epsilon = 0.1
        self.device = device

    def preprocess_state(self, state):
        if isinstance(state, dict):
            # Flatten the dictionary values into a single list
            state = list(state.values())
            state = [item for sublist in state for item in
                     (sublist if isinstance(sublist, (list, tuple, np.ndarray)) else [sublist])]

        if isinstance(state, list) or (isinstance(state, np.ndarray) and state.dtype == object):
            # Convert to a numpy array with float32 dtype
            state = np.array(state, dtype=np.float32)
        # Convert to a FloatTensor and move to the appropriate device
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)
    def select_action(self, state, evaluation=False):
        print(f"State before preprocessing: {state}, type: {type(state)}")
        state = self.preprocess_state(state)
        if not evaluation and random.random() < self.epsilon:
            action =  np.random.randint(self.action_dim)

            with torch.no_grad():
                return self.actor(state).cpu().numpy()[0]

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) < 1000:
            return
        if len(self.replay_buffer) > 10000:
            self.replay_buffer.pop(0)

        batch = random.sample(self.replay_buffer, 64)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        q_values = self.actor(state)
        q_value = q_values.gather(1, action.long().unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_q_values = self.target_q_network(next_state)
            next_q_value = next_q_values.max(1)[0]
            target_q_value = reward + (1 - done) * self.gamma * next_q_value

        loss = self.criterion(q_value, target_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if random.random() < 0.1:
            self.target_q_network.load_state_dict(self.actor.state_dict())

class DQNCooperator(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super().__init__()
        # Initialize cooperative actors
        self.actor = QNetworkBase(state_dim, action_dim).to(device)
        self.target_q_network = QNetworkBase(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.replay_buffer = []
        self.action_dim = action_dim
        self.gamma = 0.99
        self.epsilon = 0.1
        self.device = device

    def preprocess_state(self, state):
        if isinstance(state, dict):
            # Flatten the dictionary values into a single list
            state = list(state.values())
            state = [item for sublist in state for item in
                     (sublist if isinstance(sublist, (list, tuple, np.ndarray)) else [sublist])]

        if isinstance(state, list) or (isinstance(state, np.ndarray) and state.dtype == object):
            # Convert to a numpy array with float32 dtype
            state = np.array(state, dtype=np.float32)

        # Convert to a FloatTensor and move to the appropriate device
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)

    def select_action(self, state, evaluation=False):
        state = self.preprocess_state(state)
        if not evaluation and random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)

            with torch.no_grad():
                return self.actor(state).cpu().numpy()[0]

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) < 1000:
            return
        if len(self.replay_buffer) > 10000:
            self.replay_buffer.pop(0)

        batch = random.sample(self.replay_buffer, 64)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        q_values = self.actor(state)
        q_value = q_values.gather(1, action.long().unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_q_values = self.target_q_network(next_state)
            next_q_value = next_q_values.max(1)[0]
            target_q_value = reward + (1 - done) * self.gamma * next_q_value

        loss = self.criterion(q_value, target_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if random.random() < 0.1:
            self.target_q_network.load_state_dict(self.actor.state_dict())