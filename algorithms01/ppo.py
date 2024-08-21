import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCriticBase(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticBase, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.action_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_prob = torch.softmax(self.action_head(x), dim=-1)
        state_value = self.value_head(x)
        return action_prob, state_value

class PPOAdversary(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super().__init__()
        self.actor_critic = ActorCriticBase(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=3e-4)
        self.replay_buffer = []
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.device = device

    def select_action(self, state, evaluation=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_prob, _ = self.actor_critic(state)
        action = np.random.choice(len(action_prob.cpu().numpy().flatten()), p=action_prob.cpu().numpy().flatten())
        return action

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) < 1000:
            return

        state, action, reward, next_state, done = zip(*self.replay_buffer)
        self.replay_buffer = []

        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        _, next_state_values = self.actor_critic(next_state)
        _, state_values = self.actor_critic(state)

        advantages = reward + (1 - done) * self.gamma * next_state_values.squeeze() - state_values.squeeze()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        action_prob, state_values = self.actor_critic(state)
        action_prob = action_prob.gather(1, action.unsqueeze(1)).squeeze(1)
        old_action_prob = action_prob.clone().detach()

        ratio = action_prob / old_action_prob
        sur1 = ratio * advantages
        sur2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

        actor_loss = -torch.min(sur1, sur2).mean()
        critic_loss = ((reward + (1 - done) * self.gamma * next_state_values.squeeze() - state_values.squeeze()) ** 2).mean()
        loss = actor_loss + 0.5 * critic_loss - 0.01 * action_prob.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class PPOCooperator(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super().__init__()
        self.actor_critic = ActorCriticBase(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=3e-4)
        self.replay_buffer = []
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.device = device

    def select_action(self, state, evaluation=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_prob, _ = self.actor_critic(state)
        action = np.random.choice(len(action_prob.cpu().numpy().flatten()), p=action_prob.cpu().numpy().flatten())
        return action

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) < 1000:
            return

        state, action, reward, next_state, done = zip(*self.replay_buffer)
        self.replay_buffer = []

        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        _, next_state_values = self.actor_critic(next_state)
        _, state_values = self.actor_critic(state)

        advantages = reward + (1 - done) * self.gamma * next_state_values.squeeze() - state_values.squeeze()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        action_prob, state_values = self.actor_critic(state)
        action_prob = action_prob.gather(1, action.unsqueeze(1)).squeeze(1)
        old_action_prob = action_prob.clone().detach()

        ratio = action_prob / old_action_prob
        sur1 = ratio * advantages
        sur2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

        actor_loss = -torch.min(sur1, sur2).mean()
        critic_loss = ((reward + (1 - done) * self.gamma * next_state_values.squeeze() - state_values.squeeze()) ** 2).mean()
        loss = actor_loss + 0.5 * critic_loss - 0.01 * action_prob.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
