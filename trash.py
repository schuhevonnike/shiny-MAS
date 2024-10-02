import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

from scipy.constants import barrel


# Define actor network
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Actor, self).__init__()
        #self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Assuming continuous action space, which is true for simple_tag

# Define critic network
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size*3, hidden_size)  # Combined state and action
        self.fc2 = nn.Linear(hidden_size, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, state_action):
        #x = torch.cat([state, action], dim=1)  # Concatenate state and action
        x = torch.relu(self.fc1(state_action))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class MADDPGAgent:
    def __init__(self, agent_id, state_size, action_size, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=0.01):
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

        # Initialize target networks to match initial actor/critic networks
        #self._update_target_networks(tau=1.0)

        self._hard_update(self.actor, self.target_actor)
        self._hard_update(self.critic, self.target_critic)

    @staticmethod
    def _soft_update(local_model: nn.Module, target_model: nn.Module, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def _hard_update(local_model: nn.Module, target_model: nn.Module):
        target_model.load_state_dict(local_model.state_dict())

    #def act(self, state: np.ndarray, noise: float=0.1):
    #    Select an action based on the current policy
    #    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    #    with torch.no_grad():
    #       action = self.actor(state).detach().numpy()[0]
    #       action = self.actor(state).squeeze(0).numpy()
    #    self.actor.train()
    #    action += noise * np.random.randn(*action.shape)  # Add exploration noise
    #    return np.clip(action, -1, 1)  # Assuming action space [-1, 1]

    @staticmethod
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
        actions = torch.tensor(np.array([t[1] for t in experiences]), dtype=torch.float32)
        rewards = torch.tensor(np.array([t[2] for t in experiences]), dtype=torch.float32)
        next_states = torch.tensor(np.array([t[3] for t in experiences]), dtype=torch.float32)
        dones = torch.tensor(np.array([t[4] for t in experiences]), dtype=torch.bool)

        # Ensure states is a 2D tensor
        if states.dim() == 1:  # If it’s a single state
            states = states.unsqueeze(0)  # Make it a batch of one

        # Debugging: Print the agents list and the other agent ID
        print(f"Agents list: {agents}")
        print(f"Agent ID: {self.agent_id}")
        print(f"Other agent ID: {other_agent_id}")
        print(f"self.state_size: {self.state_size}, self.action_size: {self.action_size}, other_agent.action_size: {other_agent.action_size}")

        # Gather actions from other agents
        other_actions = [agents[agent_key].actor(states) for agent_key in agents if agent_key != self.agent_id]
        #other_actions = [agents[agent_key].actor(states) for agent_key in agents if agent_key != other_agent_id]

        # Make sure the actions are gathered in the correct format and concatenated in the second dimension
        if len(other_actions) > 0:
            other_actions_tensor = torch.cat(other_actions, dim=0)  # Shape: [batch_size * num_other_agents, action_size]
        else:
            raise ValueError("No other actions were gathered.")

        # Print shape for debugging
        print(f"Shape of other_actions_tensor after gathering: {other_actions_tensor.shape}")

        # Reshape to get to [batch_size, num_other_agents * action_size]
        num_other_agents = len(other_actions)  # This should give you the count of other agents
        other_actions_tensor = other_actions_tensor.view(batch_size, num_other_agents * self.action_size)

        with torch.no_grad():
            if isinstance(other_actions, list):
                if len(other_actions) == 0:
                    raise ValueError("other_actions list is empty.")
                other_actions = np.array(other_actions)  # Convert to numpy array
            elif isinstance(other_actions, torch.Tensor):
                other_actions = other_actions.detach().numpy()  # Convert to numpy array safely

        # Ensure that other_actions is not a list or an incompatible type
        if not isinstance(other_actions, np.ndarray):
            raise ValueError("other_actions must be a numpy array after conversion.")

        # Create state-action pairs for the target critic
        next_actions = self.target_actor(next_states)

        # Collect next actions from all other agents
        other_next_actions = [agents[agent_key].target_actor(next_states) for agent_key in agents if agent_key != self.agent_id]

        # Print for debugging
        print(f"Other next actions: {[action.shape for action in other_next_actions]}")

        # Ensure all actions are tensors
        assert all(isinstance(action, torch.Tensor) for action in other_next_actions), "All actions should be tensors"

        # Concatenate next_states and other_actions_tensor
        next_state_action = torch.cat([next_states, other_actions_tensor], dim=1)

        # Calculate target Q-values
        target_q = rewards + (1 - dones.to(torch.float32)) * self.gamma * self.target_critic(next_state_action)

        print(f"target_q shape: {target_q.shape}")

        # Ensure states and actions are the right shapes
        #states = states.view(batch_size, self.action_size) if states.dim() == 1 else actions
        states = states.view(batch_size, self.state_size)  # Shape: [batch_size, state_size]
        actions = actions.view(-1, 1)  # Ensure actions are 2D with shape [32, 1]
        #actions = actions.view(batch_size, self.action_size) if actions.dim() == 1 else actions
        #actions = actions.view(batch_size, self.action_size)  # Shape: [batch_size, action_size]

        # Concatenate states, actions, and other actions
        current_state_action = torch.cat([states, actions, other_actions_tensor], dim=1)

        print(f"State Size: {self.state_size}, Action Size: {self.action_size}, Other Actions Size: {other_actions_tensor.shape[1]}")
        print(f"Current State Action Shape: {current_state_action.shape}")

        # Ensure the final shape is as expected
        expected_shape = self.state_size + self.action_size + (num_other_agents * self.action_size)
        assert current_state_action.shape[1] == expected_shape, "Mismatch in current_state_action shape"

        #assert current_state_action.shape[1] == (self.state_size + self.action_size + other_actions_tensor.shape[1]), "Mismatch in current_state_action shape"

        current_q = self.critic(current_state_action)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_q, target_q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        predicted_actions = self.actor(states)
        actor_state_action = torch.cat(
            [states, predicted_actions] + [agents[agent_key].actor(states) for agent_key in agents if
                                           agent_key != other_agent_id], dim=1)
        assert actor_state_action.shape[1] == (
                    self.state_size + self.action_size + other_agent.action_size), "Mismatch in actor_state_action shape"
        actor_loss = -self.critic(actor_state_action).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self._soft_update(self.critic, self.target_critic, self.tau)
        self._soft_update(self.actor, self.target_actor, self.tau)

    def remember(self, state, action, reward, next_state, done: bool):
        # Store experience in replay buffer
        self.memory.append((state, action, reward, next_state, done))

    #def sample_memory(self, batch_size):
        # Sample a batch of experiences from memory
        #experiences = zip(*[self.memory[i] for i in np.random.choice(len(self.memory), batch_size)])
        #return experiences
    #    return random.sample(self.memory, batch_size)