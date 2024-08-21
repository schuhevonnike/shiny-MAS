import gym
import numpy as np
from gym.spaces import Box
from pettingzoo.utils import wrappers
from pettingzoo.classic import rps_v2
from pettingzoo.mpe import simple_tag_v3
import argparse

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

    def reset(self):
        # Reset the environment and return the initial observation
        obs = self.env.reset()
        self._env_done = False
        self.agent_done = {agent: False for agent in self.possible_agents}
        return self._convert_obs(obs)

    def step(self, action):    
        # Take a step in the environment.
        # Returns converted observation, reward, done flag, and info dict.
        agent = self._agent_selector
        obs, rewards, done, info = self.env.step(action, agent)
        self.agent_done[agent] = done
        self._env_done = all(self.agent_done.values())
        return self._convert_obs(obs), self._convert_reward(rewards), self._convert_done(done), info

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

class SimpleVecEnv:
    # A simple vectorized environment that contains a single environment instance.
    # This class mimics some of the functionality of stable_baselines3's DummyVecEnv.
    def __init__(self, env_fn):
        self.env = env_fn()
        self.num_envs = 1  # Only one environment in this simple implementation

    def reset(self):
        # Reset the environment and return the initial observation
        return self.env.reset()

    def step(self, actions):
        # Take a step in the environment.
        # Expects a list of actions (even though we only have one environment).
        return self.env.step(actions[0])

    def render(self, mode='human'):
        # Render the environment
        return self.env.render(mode)

    def close(self):
        # Close the environment
        return self.env.close()

class PettingZooParallelWrapper(wrappers.BaseParallelWrapper):
    def __init__(self, env_fn, num_envs):
        self.envs = [env_fn() for _ in range(num_envs)]
        super().__init__(self.envs[0])  # Initialize with a single env for metadata
        self.num_envs = num_envs
        
        # Store action and observation spaces
        self._action_spaces = {agent: self.envs[0].action_space(agent) for agent in self.envs[0].possible_agents}
        self._observation_spaces = {agent: self.envs[0].observation_space(agent) for agent in self.envs[0].possible_agents}

    def reset(self):
        return [env.reset() for env in self.envs]

    def step(self, actions):
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        obs, rewards, done, infos = zip(*results)
        return list(obs), list(rewards), list(done), list(infos)

    def action_space(self, agent):
        return self._action_spaces[agent]

    def observation_space(self, agent):
        return self._observation_spaces[agent]

def make_env1(individual=True, num_envs=1):
    def env_fn():
        if individual:
            return simple_tag_v3.parallel_env()
        else:
            return simple_tag_v3.parallel_env(max_cycles=50)

    return PettingZooParallelWrapper(env_fn, num_envs)

def make_env2(individual=True, num_envs=1):
    def env_fn():
        if individual:
            return rps_v2.parallel_env()
        else:
            return rps_v2.parallel_env(num_actions=5)

    return PettingZooParallelWrapper(env_fn, num_envs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PettingZoo Environment Configuration')
    parser.add_argument('--num_envs', type=int, default=4, help='Number of parallel environments')
    args = parser.parse_args()

    # Example usage
    individual_env1 = make_env1(individual=True, num_envs=args.num_envs)
    cooperative_env1 = make_env1(individual=False, num_envs=args.num_envs)
    
    individual_env2 = make_env2(individual=True, num_envs=args.num_envs)
    cooperative_env2 = make_env2(individual=False, num_envs=args.num_envs)
    
    print("Environments created successfully.")
