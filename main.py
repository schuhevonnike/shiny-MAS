import torch
import pandas as pd
from environments.pettingzoo_env import PettingZooEnv, make_env1, make_env2
from algorithms01.dqn import DQN as DQN_online
from algorithms01.ppo import PPO as PPO_online
from algorithms01.maddpg import MADDPG as MADDPG_online
from algorithms01.sac import SAC as SAC_online
from algorithms02.dqn import DQN as DQN_own
from algorithms02.ppo import PPO as PPO_own
from algorithms02.maddpg import MADDPG as MADDPG_own
from algorithms02.sac import SAC as SAC_own
from utils.training import train
from utils.evaluation import evaluate
from utils.metrics import calculate_metrics
import os
import gymnasium as gym
import argparse

# Main experiment parameters
env_names = ['make_env1', 'make_env2']  # Choose the environment to run
algorithms = ['DQN', 'PPO', 'MADDPG', 'SAC']
settings = ['individual', 'cooperative']
sources = ['online', 'own']
# Cuda cores schneller
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_experiment(env_name, algorithm, individual, source, device, num_envs):
    if env_name == 'make_env1':
        env = make_env1(individual=individual, num_envs=num_envs)
    elif env_name == 'make_env2':
        env = make_env2(individual=individual, num_envs=num_envs)
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    agents = {}  # Initialize the agents dictionary

    # Iterate over all agents to get action and observation spaces
    for agent_id in env.envs[0].possible_agents:
        action_space = env.action_space(agent_id)
        observation_space = env.observation_space(agent_id)

        print(f"Agent: {agent_id}, Action space: {action_space}, Observation space: {observation_space}")
        if isinstance(action_space, gym.spaces.Box):
            print(f"Action space shape: {action_space.shape}")
            print(f"Action space low: {action_space.low}")
            print(f"Action space high: {action_space.high}")

        state_dim = observation_space.shape[0]
        if isinstance(action_space, gym.spaces.Discrete):
            action_dim = action_space.n
        elif isinstance(action_space, gym.spaces.Box):
            action_dim = action_space.shape[0] if action_space.shape else 1
        else:
            raise ValueError(f"Unsupported action space type: {type(action_space)}")

        max_action = 1 if isinstance(action_space, gym.spaces.Discrete) else action_space.high[0]

        if source == 'online':
            if algorithm == 'DQN':
                agents[agent_id] = DQN_online(state_dim, action_dim, device)
            elif algorithm == 'PPO':
                agents[agent_id] = PPO_online(state_dim, action_dim, device)
            elif algorithm == 'MADDPG':
                agents[agent_id] = MADDPG_online(state_dim, action_dim, len(env.envs[0].possible_agents), max_action, device)
            elif algorithm == 'SAC':
                agents[agent_id] = SAC_online(state_dim, action_dim, max_action, device)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
        elif source == 'own':
            if algorithm == 'DQN':
                agents[agent_id] = DQN_own(state_dim, action_dim, device)
            elif algorithm == 'PPO':
                agents[agent_id] = PPO_own(state_dim, action_dim, device)
            elif algorithm == 'MADDPG':
                agents[agent_id] = MADDPG_own(state_dim, action_dim, len(env.envs[0].possible_agents), max_action, device)
            elif algorithm == 'SAC':
                agents[agent_id] = SAC_own(state_dim, action_dim, max_action, device)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

    # Training phase
    train(env, agents, algorithm, source, num_episodes=1000)

    # Evaluation phase
    mean_reward, std_reward = evaluate(env, agents, num_episodes=100)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

    return env, mean_reward, std_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run RL experiments')
    parser.add_argument('--num_envs', type=int, default=4, help='Number of parallel environments')
    args = parser.parse_args()

    # Directory to save results
    os.makedirs('results', exist_ok=True)

    # Running experiments for each combination of algorithm, setting, and source
    for source in sources:
        for setting in settings:
            # Initialize empty array for storing the results
            results = []
            for algo in algorithms:
                individual = (setting == 'individual')
                for env_name in env_names:
                    env, mean_reward, std_reward = run_experiment(env_name, algo, individual=individual, source=source, device=device, num_envs=args.num_envs)
                    results.append({'environment': env_name, 'algorithm': algo, 'setting': setting, 'source': source, 'mean_reward': mean_reward, 'std_reward': std_reward})
            # Saving the results to a CSV file for each setting and source
            filename = f'results/comparison_results_{source}_{setting}.csv'
            results_df = pd.DataFrame(results)
            results_df.to_csv(filename, index=False)
