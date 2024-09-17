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

class DQNAgent:
    def __init__(self, state_size, action_size, cooperative=False, learning_rate=1e-3, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, min_epsilon=0.01):
        self.input_dim = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.memory = deque(maxlen=2000)
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        #self.cooperative = cooperative

    # Manually added method to reshape tensors to avoid DimensionMismatch (old, misfunctional version)
    #def reshape_tensor(self, tensor, desired_shape):
    #    if tensor.shape != desired_shape:
    #        if tensor.shape[1] < desired_shape[1]:
    #            padding_size = desired_shape[1] - tensor.shape[1]
    #            padding = torch.zeros(tensor.shape[0], padding_size, dtype=tensor.dtype)
    #            tensor = torch.cat([tensor, padding], dim=1)
    #        elif tensor.shape[1] > desired_shape[1]:
    #            tensor = tensor[:, :desired_shape[1]]
    #    return tensor

    #def reshape_tensor(self, tensor, desired_shape):
        # Ensure that the number of dimensions is the same
    #    if tensor.dim() != len(desired_shape):
    #        raise ValueError(
    #            f"Tensor has {tensor.dim()} dimensions but desired shape requires {len(desired_shape)} dimensions.")

        # Process each dimension independently
    #    for i in range(len(desired_shape)):
    #        if tensor.shape[i] < desired_shape[i]:
    #            # Padding for the current dimension
    #            padding_size = desired_shape[i] - tensor.shape[i]
    #            pad_shape = list(tensor.shape)
    #            pad_shape[i] = padding_size
    #            padding = torch.zeros(*pad_shape, dtype=tensor.dtype)
    #            tensor = torch.cat([tensor, padding], dim=i)
    #        elif tensor.shape[i] > desired_shape[i]:
    #            # Trimming for the current dimension
    #            slices = [slice(None)] * len(tensor.shape)
    #            slices[i] = slice(0, desired_shape[i])
    #            tensor = tensor[tuple(slices)]

    #    return tensor

    # New, reformulated reshape_tensor() method
    def reshape_tensor(self, tensor, desired_shape):
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

    # Old, initial act() method:
    #def act(self, state, other_agents=None):
    #    if np.random.rand() <= self.epsilon:
    #        return random.randrange(self.action_size)
    #    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    #    state = self.reshape_tensor(state, (1, self.input_dim))
    #    q_values = self.model(state)

    #    if self.cooperative and other_agents:
    #        combined_q_values = q_values.clone()
    #        for agent in other_agents:
    #            combined_q_values += agent.model(state)
    #        combined_q_values /= (1 + len(other_agents))
    #        return torch.argmax(combined_q_values).item()
    #    else:
    #        return torch.argmax(q_values).item()

    #Adjusted act() method:
    def act(self, state, other_agents=None):
        # Ensure action is within the valid range
        action = random.randrange(self.action_size)
        #Epsilon-greedy
        if np.random.rand() > self.epsilon:
            state = torch.tensor(state, dtype=torch.float32).clone().detach().unsqueeze(0)
            #state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            state = self.reshape_tensor(state, (1, self.input_dim))
            q_values = self.model(state)
            # if-check for cooperative behaviour, needs fine-tuning
            #if self.cooperative and other_agents:
            #    combined_q_values = q_values.clone()
            #    for agent in other_agents:
            #        combined_q_values += agent.model(state)
            #    combined_q_values /= (1 + len(other_agents))
            #    action = torch.argmax(combined_q_values).item()
            #else:
            action = torch.argmax(q_values).item()
        # Add debug prints to ensure action is valid
        #print(f"Selected action: {action}")
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            state = self.reshape_tensor(state, (1, self.input_dim))
            #print(f"Shape of input tensor 'state': {state.shape}")

            #next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            next_state = next_state.clone().detach()

            next_state = self.reshape_tensor(next_state, (1, self.input_dim))
            #print(f"Shape of input tensor 'next_state': {next_state.shape}")

            assert state.shape[1] == self.input_dim, f"State dimension mismatch: {state.shape[1]} vs {self.input_dim}"
            assert next_state.shape[1] == self.input_dim, f"Next state dimension mismatch: {next_state.shape[1]} vs {self.input_dim}"

            #reward = reward.clone().detach().float().unsqueeze(0)
            reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
            #done = done.clone().detach().float().unsqueeze(0)
            done = torch.tensor(done, dtype=torch.float32).unsqueeze(0)
            target = reward.clone()

            if not done.item():
                with torch.no_grad():
                    next_state_value = self.model(next_state).max(1)[0]
                    #print(f"Next state value shape: {next_state_value.shape}")
                target = reward + (1 - done) * self.gamma * next_state_value
                #target += self.gamma * torch.max(next_state_value)

            output = self.model(state)[0, action].unsqueeze(0)
            # Fix target shape if it is a scalar or has incompatible shape
            #if target.dim() == 1:  # If target is a vector
            #    target = target.expand(output.shape)
            #elif target.dim() == 0:  # If target is a scalar
            #    target = torch.full_like(output, target.item())

            # target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
            #print(f"Shape of target tensor: {target.shape}")

            # Ensure target and output are of the same shape
            #assert output.shape == target.shape, f"Output shape {output.shape} does not match target shape {target.shape}"

            # Debug prints to ensure no NaNs or invalid values
            if torch.isnan(output).any() or torch.isnan(target).any():
                print("NaN detected in output or target!")
                continue

            # Loss calculation
            loss = self.criterion(output, target)

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Epsilon decay
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay
