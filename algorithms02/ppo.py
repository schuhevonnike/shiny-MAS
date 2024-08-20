import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class PPO:
    def __init__(self, state_dim, action_dim, device, learning_rate=0.0003, gamma=0.99, epsilon=0.2, batch_size=64):
        self.device = device
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(device)
        self.value_net = ValueNetwork(state_dim).to(device)
        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.replay_buffer = []

        def select_action(self, state):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action_probs = self.policy_net(state)
            action = np.random.choice(len(action_probs.cpu().numpy()[0]), p=action_probs.cpu().numpy()[0])
            return action

        def store_transition(self, transition):
            self.replay_buffer.append(transition)

        def update(self):
            if len(self.replay_buffer) < self.batch_size:
                return

            states, actions, rewards, next_states, dones = zip(*self.replay_buffer)
            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).to(device)

            old_action_probs = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            returns = self.compute_returns(rewards, dones, next_states)
            advantages = returns - self.value_net(states).squeeze(1)

            for _ in range(10):  # multiple updates per batch
                new_action_probs = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                ratios = new_action_probs / old_action_probs
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(self.value_net(states).squeeze(1), returns)
                
                self.optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                self.optimizer.step()

            self.replay_buffer = []

        def compute_returns(self, rewards, dones, next_states):
            returns = []
            G = 0
            for reward, done, next_state in zip(reversed(rewards), reversed(dones), reversed(next_states)):
                if done:
                    G = 0
                G = reward + self.gamma * G
                returns.insert(0, G)
            returns = torch.FloatTensor(returns).to(device)
            return returns

        def save(self, filename):
            torch.save(self.policy_net.state_dict(), filename)

        def load(self, filename):
            self.policy_net.load_state_dict(torch.load(filename))
