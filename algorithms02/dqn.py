import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQN:
    def __init__(self, state_dim, action_dim, device, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, buffer_size=10000, batch_size=64):
        self.device = device 
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = deque(maxlen=buffer_size)
        self.update_target_network()

        def update_target_network(self):
            self.target_network.load_state_dict(self.q_network.state_dict())

        def select_action(self, state, evaluation=False):
            if evaluation or np.random.rand() > self.epsilon:
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = self.q_network(state).argmax().item()
            else:
                action = np.random.randint(self.action_dim)
            return action

        def store_transition(self, state, action, reward, next_state, done):
            self.replay_buffer.append((state, action, reward, next_state, done))

        def update(self):
            if len(self.replay_buffer) < self.batch_size:
                return

            batch = random.sample(self.replay_buffer, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).to(device)

            q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                max_next_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

            loss = nn.MSELoss()(q_values, target_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay

        def save(self, filename):
            torch.save(self.q_network.state_dict(), filename)

        def load(self, filename):
            self.q_network.load_state_dict(torch.load(filename))
            self.update_target_network()
