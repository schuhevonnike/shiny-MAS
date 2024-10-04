import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# learning_rate == Beta (ß) is a hyperparameter
class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=1e-3, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.999, min_epsilon=0.01, target_update_freq=1000):
        self.input_dim = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.memory = deque(maxlen=10000)
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_steps = 0
        self.target_update_freq = target_update_freq
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    @staticmethod
    # New, reformulated reshape_tensor() method
    def reshape_tensor(tensor, desired_shape):
        if len(desired_shape) == 2 and tensor.dim() > 2:
            # Flatten the tensor except for the batch dimension
            tensor = tensor.view(tensor.size(0), -1)

        if tensor.shape != desired_shape:
            if tensor.shape[1] < desired_shape[1]:
                padding_size = desired_shape[1] - tensor.shape[1]
                padding = torch.zeros(tensor.shape[0], padding_size, dtype=tensor.dtype)
                tensor = torch.cat([tensor, padding], dim=1)
            elif tensor.shape[1] > desired_shape[1]:
                tensor = tensor[:, :desired_shape[1]]
        return tensor

    def update(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        states = torch.tensor(np.array([t[0] for t in minibatch]), dtype=torch.float32)
        actions = torch.tensor(np.array([t[1] for t in minibatch]), dtype=torch.long)
        rewards = torch.tensor(np.array([t[2] for t in minibatch]), dtype=torch.float32)
        next_states = torch.tensor(np.array([t[3] for t in minibatch]), dtype=torch.float32)
        dones = torch.tensor(np.array([t[4] for t in minibatch]), dtype=torch.bool)

        # Compute Q(s_t, a)
        state_action_values = self.model(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            # Use target network for next state value estimation
            next_state_values = self.target_model(next_states).max(1)[0]
            # Set next_state_values to zero where done is True
            next_state_values[dones] = 0.0
            expected_state_action_values = rewards + (self.gamma * next_state_values)

        # Loss calculation
        loss = self.criterion(state_action_values.squeeze(), expected_state_action_values)
        # print(loss, state_action_values.mean(), expected_state_action_values.mean())
        #loss = self.criterion(output, target)

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        #if self.epsilon > self.min_epsilon:
        #    self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        #print(self.epsilon)
        # Update the target network
        self.update_steps += 1
        if self.update_steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done: bool):
        self.memory.append((state, action, reward, next_state, done))

