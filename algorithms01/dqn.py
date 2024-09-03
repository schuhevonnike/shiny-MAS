import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)  # Ensure state_size matches the input
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        # Feed forward network using ReLU activation function
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, action_size, cooperative=False, learning_rate=1e-3, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, min_epsilon=0.01):
        self.input_dim = state_size  # Define input_dim here
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.memory = deque(maxlen=2000)
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.cooperative = cooperative  # New attribute to track cooperative behavior

    def act(self, state, other_agents=None):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.model(state)

        if self.cooperative and other_agents:
            # Cooperative decision making: example implementation
            # Average Q-values with other agents (simple example)
            combined_q_values = q_values.clone()
            for agent in other_agents:
                combined_q_values += agent.model(state)
            combined_q_values /= (1 + len(other_agents))
            return torch.argmax(combined_q_values).item()
        else:
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:

            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Shape: (1, state_size)
            state = state.clone().detach().float()
            print(f"Shape of input tensor 'state': {state.shape}")
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)  # Shape: (1, state_size)
            #next_state = next_state.clone().detach().float()
            print(f"Shape of input tensor 'next_state': {next_state.shape}")

            # Ensure shapes match the expected dimensions
            assert state.shape[1] == self.input_dim, f"State dimension mismatch: {state.shape[1]} vs {self.input_dim}"
            assert next_state.shape[1] == self.input_dim, f"Next state dimension mismatch: {next_state.shape[1]} vs {self.input_dim}"

            # Convert reward and done to tensors
            reward = torch.tensor(reward, dtype=torch.float32)
            reward = reward.clone().detach().float()
            done = torch.tensor(done, dtype=torch.float32)
            done = done.clone().detach().float()
            target = reward

            if not done:
                with torch.no_grad():
                    next_state_value = self.model(next_state)
                    print(f"Next state value shape: {next_state_value.shape}")
                target += self.gamma * torch.max(next_state_value)

            output = self.model(state)[0, action]

            # Ensure target is a tensor with the same shape as output
            target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
            target = target.clone().detach().float().unsqueeze(0)
            print(f"Shape of target tensor: {target.shape}")

            # Loss calculation
            loss = self.criterion(output, target)

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Epsilon decay
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay
