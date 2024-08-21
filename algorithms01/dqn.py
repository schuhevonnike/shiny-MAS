import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from pettingzoo.utils import agent_selector

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
        self.actor = QNetworkBase(state_dim, action_dim).to(device)
        self.target_q_network = QNetworkBase(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.replay_buffer = []
        self.action_dim = action_dim
        self.gamma = 0.99
        self.epsilon = 0.1
        self.device = device

    def select_action(self, state, evaluation=False):
        if not evaluation and random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        # If state is a dictionary, convert it to a list of values
        if isinstance(state, dict):
            state = list(state.values())
            # If the values are lists or arrays, flatten them
            state = [item for sublist in state for item in
                     (sublist if isinstance(sublist, (list, tuple, np.ndarray)) else [sublist])]

        # Ensure state is a numpy array of the correct dtype
        if isinstance(state, list):
            state = np.array(state, dtype=np.float32)
        elif isinstance(state, np.ndarray) and state.dtype == object:
            state = np.array(state.tolist(), dtype=np.float32)
            return state

        # Folgende Zeile schmeißt nen Error:
        #state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if random.random() < self.epsilon:
            action = np.random.uniform(-1, 1, size=self.actor.fc3.out_features)
        else:
            with torch.no_grad():
                action = self.actor(state).cpu().numpy()[0]
        return action

        # Ensure state length matches the expected length (e.g., 16)
        expected_length = 16  # Replace with the actual expected length
        current_length = len(state)

        if current_length < expected_length:
            # Pad state with zeros if it's shorter than expected
            state = np.pad(state, (0, expected_length - current_length), 'constant')
        elif current_length > expected_length:
            # Trim state if it's longer than expected
            state = state[:expected_length]

        # Convert the state to a PyTorch tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Use the model to select the action
        with torch.no_grad():
            return self.actor(state).argmax().item()

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

        q_values = self.actor(state).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target_q_values = reward + (1 - done) * self.gamma * self.target_q_network(next_state).max(1)[0]
        loss = self.criterion(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if random.random() < 0.1:
            self.target_q_network.load_state_dict(self.actor.state_dict())

class DQNCooperator(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super().__init__()
        self.actor = QNetworkBase(state_dim, action_dim).to(device)
        self.target_q_network = QNetworkBase(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.replay_buffer = []
        self.action_dim = action_dim
        self.gamma = 0.99
        self.epsilon = 0.1
        self.device = device

    def select_action(self, state, evaluation=False):
        if not evaluation and random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        # If state is a dictionary, convert it to a list of values
        if isinstance(state, dict):
            state = list(state.values())
            # If the values are lists or arrays, flatten them
            state = [item for sublist in state for item in
                     (sublist if isinstance(sublist, (list, tuple, np.ndarray)) else [sublist])]
            return state
        # Ensure state is a numpy array of the correct dtype
        if isinstance(state, list):
            state = np.array(state, dtype=np.float32)
        elif isinstance(state, np.ndarray) and state.dtype == object:
            state = np.array(state.tolist(), dtype=np.float32)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if random.random() < self.epsilon:
            action = np.random.uniform(-1, 1, size=self.actor.fc3.out_features)
        else:
            with torch.no_grad():
                action = self.actor(state).cpu().numpy()[0]
        return action

        # Ensure state length matches the expected length (e.g., 16)
        expected_length = 16  # Replace with the actual expected length
        current_length = len(state)

        if current_length < expected_length:
            # Pad state with zeros if it's shorter than expected
            state = np.pad(state, (0, expected_length - current_length), 'constant')
        elif current_length > expected_length:
            # Trim state if it's longer than expected
            state = state[:expected_length]

        # Convert the state to a PyTorch tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Use the model to select the action
        with torch.no_grad():
            return self.actor(state).argmax().item()

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

        q_values = self.actor(state).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target_q_values = reward + (1 - done) * self.gamma * self.target_q_network(next_state).max(1)[0]
        loss = self.criterion(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if random.random() < 0.1:
            self.target_q_network.load_state_dict(self.actor.state_dict())