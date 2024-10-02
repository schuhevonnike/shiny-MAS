import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

from scipy.constants import barrel
from torch.onnx.symbolic_opset11 import unsqueeze

# Define actor network
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)  # Discrete action space

# Define critic network
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)  # Combined state and action
        self.fc2 = nn.Linear(hidden_size, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state, action):
        # Ensure that both state and action are 2D tensors with batch size as the first dimension
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        x = torch.cat([state, action], dim=1)  # Concatenate state and action
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class MADDPGAgent:
    def __init__(self, agent_id, state_size, action_size, actor_lr=1e-3, critic_lr=1e-3, gamma=0.9999, tau=0.0001): # Tau for soft updating the network
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size

        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size, action_size)  # Assuming centralized critic with both agents' states/actions
        self.target_actor = Actor(state_size, action_size)
        self.target_critic = Critic(state_size, action_size)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau  # Soft update parameter
        self.memory = deque(maxlen=10000)

        self._hard_update(self.actor, self.target_actor)
        self._hard_update(self.critic, self.target_critic)

    #def act(self, state: np.ndarray, noise: float=0.1):
    #    Select an action based on the current policy
    #    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    #    with torch.no_grad():
    #       action = self.actor(state).detach().numpy()[0]
    #       action = self.actor(state).squeeze(0).numpy()
    #    self.actor.train()
    #    action += noise * np.random.randn(*action.shape)  # Add exploration noise
    #    return np.clip(action, -1, 1)  # Assuming action space [-1, 1]

    def update(self, other_agent_id, agents, batch_size=32):
        # Only update if we have enough samples in memory
        if len(self.memory) < batch_size:
            return  # Not enough samples to perform an update

        # Ensure the actual agent object is passed
        other_agent = agents[other_agent_id]

        # Update actor and critic networks
        experiences = random.sample(self.memory, batch_size)

        # Convert experience tuples to tensors
        states = torch.tensor(np.array([t[0] for t in experiences]), dtype=torch.float32)
        #actions = torch.tensor(np.array([t[1] for t in experiences]), dtype=torch.float32)
        actions = torch.tensor(np.array([t[1] for t in experiences]), dtype=torch.long)
        rewards = torch.tensor(np.array([t[2] for t in experiences]), dtype=torch.float32)
        next_states = torch.tensor(np.array([t[3] for t in experiences]), dtype=torch.float32)
        dones = torch.tensor(np.array([t[4] for t in experiences]), dtype=torch.bool)

        #print(type(actions))
        # Debugging: Print the agents list and the other agent ID
        #print(f"Agents list: {agents}")
        #print(f"Agent ID: {self.agent_id}")
        #print(f"Other agent ID: {other_agent_id}")
        #print(f"self.state_size: {self.state_size}, self.action_size: {self.action_size}, other_agent.action_size: {other_agent.action_size}")
        # Check the shape of actions before reshaping
        #print(f"Original actions shape: {actions.shape}")  # This should be [batch_size, action_size] or [32, action_size]

        # Gather next agents from the current agent
        next_actions = torch.cat([agents[agent].target_actor(next_states) for agent in agents if agent == self.agent_id])
        #print(f"Next actions shape after concatenating: {next_actions.shape}")

        # Collect next actions from all other agents
        other_next_actions = [agents[agent_key].target_actor(next_states) for agent_key in agents if agent_key != self.agent_id]
        #print(f"Other next actions shape before stack: {[action.shape for action in other_next_actions]}")
        #print(type(other_next_actions))
        if len(other_next_actions) > 0:
            #other_next_actions_tensor = torch.cat(other_next_actions, dim=1)  # Concatenate actions from other agents
            other_next_actions_tensor = torch.stack(other_next_actions, dim=1)  # Concatenate actions from other agents
        else:
            raise ValueError("No other actions were gathered.")

        # Print for debugging
        #print(f"Other next actions shape after stack: {other_next_actions_tensor.shape}")

        # Concatenate next_states and other_actions_tensor
        #next_state_action = torch.cat([next_states, next_actions], dim=1)
        #print(next_state_action.shape)

        #print(next_actions.shape)
        #print(next_states.shape)

        # Calculate target Q-values
        #print(rewards.shape)
        #print(dones.shape)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)
        #print(rewards.shape)
        #print(dones.shape)
        target_q = rewards + (1 - dones.to(torch.float32)) * self.gamma * self.target_critic(next_states, next_actions)

        # Concatenate states, actions, and other actions
        #current_state_action = torch.cat([states, actions], dim=1)

        #print(f"States shape: {states.shape}")
        #print(f"Actions shape: {actions.shape}")
        #current_q = self.critic(current_states_actions)
        current_q = self.critic(states, actions)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_q, target_q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        predicted_actions = self.actor(states)
        #print(f"Predicted actions shape: {predicted_actions.shape}")
        #actor_state_action = torch.cat([states, predicted_actions] + [agents[agent_key].actor(states) for agent_key in agents if agent_key != other_agent_id], dim=1)

        #actor_loss = -self.critic(actor_state_action).mean()
        actor_loss = -self.critic(states, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self._soft_update(self.critic, self.target_critic, self.tau)
        self._soft_update(self.actor, self.target_actor, self.tau)

    def remember(self, state, action, reward, next_state, done: bool):
        # Store experience in replay buffer
        self.memory.append((state, action, reward, next_state, done))

    @staticmethod
    def _soft_update(local_model: nn.Module, target_model: nn.Module, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def _hard_update(local_model: nn.Module, target_model: nn.Module):
        target_model.load_state_dict(local_model.state_dict())

    #def sample_memory(self, batch_size):
        # Sample a batch of experiences from memory
        #experiences = zip(*[self.memory[i] for i in np.random.choice(len(self.memory), batch_size)])
        #return experiences
    #    return random.sample(self.memory, batch_size)