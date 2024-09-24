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
    def __init__(self, state_size, action_size, cooperative=False, learning_rate=1e-4, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.9999, min_epsilon=0.01, target_update_freq=1000):
        self.input_dim = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.memory = deque(maxlen=200_000)
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_steps = 0
        self.target_update_freq = target_update_freq
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

        loss = self.criterion(state_action_values.squeeze(), expected_state_action_values)
        #print(loss, state_action_values.mean(), expected_state_action_values.mean())
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        # Update the target network
        self.update_steps += 1
        if self.update_steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())