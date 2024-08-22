import gym
import numpy as np
from gym.spaces import Box
from pettingzoo.utils import wrappers, agent_selector
from pettingzoo.mpe import simple_tag_v3

class PettingZooEnv(gym.Env):
    # A wrapper class for PettingZoo environments to make them compatible with gym interface.
    def __init__(self, env):
        self.env = env
        self.possible_agents = self.env.possible_agents
        self.agent_idx = {agent: idx for idx, agent in enumerate(self.possible_agents)}
        self.observation_space = self.env.observation_space(self.possible_agents[0])
        self.action_space = self.env.action_space(self.possible_agents[0])
        self._agent_selector = self.env.agent_selector
        self._env_done = False
        self.n_agents = len(self.possible_agents)
        self.agent_done = {agent: False for agent in self.possible_agents}

    def reset(self, seed=None, options=None):
        # Reset the environment and return the initial observation
        #obs = self.env.reset()
        #self._env_done = False
        #self.agent_done = {agent: False for agent in self.possible_agents}
        #return self._convert_obs(obs)
        pass

    def step(self, action):
        # Take a step in the environment.
        # Returns converted observation, reward, done flag, and info dict.
        agent = self._agent_selector
        if agent not in self.possible_agents:
            raise ValueError(f"Agent {agent} not found in possible agents {self.possible_agents}")

        result = self.env.step(action, agent)
        if not isinstance(result, tuple) or len(result) != 4:
            raise ValueError(f"Invalid step result: {result}. Expected a tuple of (obs, reward, done, info).")

        obs, reward, done, info = result
        self.agent_done[agent] = done
        self._env_done = all(self.agent_done.values())
        return self._convert_obs(obs), self._convert_reward(reward), self._convert_done(done), info

    def _convert_obs(self, obs):
        # Convert observation to a dict with agent index as key
        return {self.agent_idx[self._agent_selector]: obs}

    def _convert_reward(self, reward):
        # Convert reward to a dict with agent index as key
        return {self.agent_idx[self._agent_selector]: reward}

    def _convert_done(self, done):
        # Convert done flag to a dict with agent index as key
        return {self.agent_idx[self._agent_selector]: done}

    def render(self, mode='human'):
        # Render the environment
        self.env.render(mode=mode)

    def close(self):
        # Close the environment
        self.env.close()

    def seed(self, seed=None):
        # Set the seed for the environment
        self.env.seed(seed)


# From pettingzoo.farama.org: Usage of simple_tag_v3 in AEC environments (classic agent environment cycles)
'''
env = simple_tag_v3.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()

    env.step(action)
env.close()
'''

def create_simple_tag_v3(num_good, num_adversaries, num_obstacles, max_cycles, continuous_actions):
    env = simple_tag_v3.env(render
    return env
