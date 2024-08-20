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
        x = torch.ReLu(self.fc1(state))
        x = torch.ReLU(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.ReLU(self.fc1(x))
        x = torch.ReLU(self.fc2(x))
        return self.fc3(x)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state):
        x = torch.ReLU(self.fc1(state))
        x = torch.ReLU(self.fc2(x))
        return self.fc3(x)

class SAC:
    def __init__(self, state_dim, action_dim, max_action, device, learning_rate=0.0003, gamma=0.99, tau=0.005, alpha=0.2):
        self.device = device
        self.actor = ActorNetwork(state_dim, action_dim, max_action).to(device)
        self.critic_1 = CriticNetwork(state_dim, action_dim).to(device)
        self.critic_2 = CriticNetwork(state_dim, action_dim).to(device)
        self.value = ValueNetwork(state_dim).to(device)
        self.target_value = ValueNetwork(state_dim).to(device)
        self.target_value.load_state_dict(self.value.state_dict())
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.optimizer_critic_1 = optim.Adam(self.critic_1.parameters(), lr=learning_rate)
        self.optimizer_critic_2 = optim.Adam(self.critic_2.parameters(), lr=learning_rate)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=learning_rate)
        self.replay_buffer = deque(maxlen=100000)

        def select_action(self, state):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = self.actor(state)
            return action.cpu().numpy()[0]

        def store_transition(self, state, action, reward, next_state, done):
            self.replay_buffer.append((state, action, reward, next_state, done))

        def update(self):
            if len(self.replay_buffer) < 64:
                return

            batch = random.sample(self.replay_buffer, 64)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(states).to(device)
            actions = torch.FloatTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            done = torch.FloatTensor(done).to(device)

            with torch.no_grad():
                next_actions = self.actor(next_states)
                target_values = self.target_value(next_states)
                target_q_values_1 = self.critic_1(next_states, next_actions)
                target_q_values_2 = self.critic_2(next_states, next_actions)
                target_q_values = torch.min(target_q_values_1, target_q_values_2) - self.alpha * target_values

            q_values_1 = self.critic_1(states, actions)
            q_values_2 = self.critic_2(states, actions)
            critic_loss_1 = nn.MSELoss()(q_values_1, rewards + self.gamma * target_q_values * (1 - dones))
            critic_loss_2 = nn.MSELoss()(q_values_2, rewards + self.gamma * target_q_values * (1 - dones))

            values = self.value(states)
            value_loss = nn.MSELoss()(values, target_q_values)

            policy_loss = (self.alpha * values - self.critic_1(states, self.actor(states))).mean()

            self.optimizer_critic_1.zero_grad()
            critic_loss_1.backward()
            self.optimizer_critic_1.step()

            self.optimizer_critic_2.zero_grad()
            critic_loss_2.backward()
            self.optimizer_critic_2.step()

            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

            self.optimizer_actor.zero_grad()
            policy_loss.backward()
            self.optimizer_actor.step()

            for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        def save(self, filename):
            torch.save(self.actor.state_dict(), filename)

        def load(self, filename):
            self.actor.load_state_dict(torch.load(filename))
