import torch
import pandas as pd
from environments.pettingzoo_env import PettingZooEnv, make_env1, make_env2
from algorithms01.dqn import DQNAdversary as DQNAdversary_01, DQNCooperator as DQNCooperator_01
from algorithms01.ppo import PPOAdversary as PPOAdversary_01, PPOCooperator as PPOCooperator_01
from algorithms01.maddpg import MADDPGAdversary as MADDPGAdversary_01, MADDPGCooperator as MADDPGCooperator_01
from algorithms01.sac import SAC as SAC_01
from algorithms02.dqn import DQNAdversary as DQNAdversary_02, DQNCooperator as DQNCooperator_02
from algorithms02.ppo import PPOAdversary as PPOAdversary_02, PPOCooperator as PPOCooperator_02
from algorithms02.maddpg import MADDPG as MADDPG_02
from algorithms02.sac import SAC as SAC_02
from utils.training import train
from utils.evaluation import evaluate
from utils.metrics import calculate_metrics
import os
import gymnasium as gym
import argparse

# Main experiment parameters
env_names = ['make_env1', 'make_env2']  # Choose the environment to run
algorithms = ['DQN', 'PPO', 'MADDPG', 'SAC']
agent_types = ['adversary', 'cooperator']
sources = ['01', '02']

# Cuda cores are better suited for such tasks, so ideally, we want to run the program on a cuda. If not possible, use the standard CPU.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_experiment(env_name, algorithm, num_envs):
    if env_name == 'make_env1':
        env = make_env1(num_envs=num_envs)
    elif env_name == 'make_env2':
        env = make_env2(num_envs=num_envs)
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    # Initialize separate dictionaries for adversary and cooperator agents
    adversary_agents = {}
    cooperator_agents = {}

    # Iterate over all agents to get action and observation spaces
    for agent_id in env.envs[0].possible_agents:
        action_space = env.action_space(agent_id)
        observation_space = env.observation_space(agent_id)

        if isinstance(action_space, gym.spaces.Discrete):
            action_dim = action_space.n
        elif isinstance(action_space, gym.spaces.Box):
            action_dim = action_space.shape[0] if action_space.shape else 1
        else:
            raise ValueError(f"Unsupported action space type: {type(action_space)}")
        state_dim = observation_space.shape[0]

        max_action = 1 if isinstance(action_space, gym.spaces.Discrete) else action_space.high[0]

        if 'adversary' in agent_id:
            agent_type = 'adversary'
        else:
            agent_type = 'cooperator'

        agents = adversary_agents if agent_type == 'adversary' else cooperator_agents

        if source == 'online':
            if algorithm == 'DQN':
                agents[agent_id] = DQNAdversary_01(state_dim, action_dim, device) if agent_type == 'adversary' else DQNCooperator_01(state_dim, action_dim, device)
            elif algorithm == 'PPO':
                agents[agent_id] = PPOAdversary_01(state_dim, action_dim, device) if agent_type == 'adversary' else PPOCooperator_01(state_dim, action_dim, device)
            elif algorithm == 'MADDPG':
                agents[agent_id] = MADDPGAdversary_01(state_dim, action_dim, device) if agent_type == 'adversary' else MADDPGCooperator_01(state_dim, action_dim, device)
            else:
                agents[agent_id] = SAC_01(state_dim, action_dim, device)
        elif source == 'own':
            if algorithm == 'DQN':
                agents[agent_id] = DQNAdversary_02(state_dim, action_dim, device) if agent_type == 'adversary' else DQNCooperator_02(state_dim, action_dim, device)
            elif algorithm == 'PPO':
                agents[agent_id] = PPOAdversary_02(state_dim, action_dim, device) if agent_type == 'adversary' else PPOCooperator_02(state_dim, action_dim, device)
            elif algorithm == 'MADDPG':
                agents[agent_id] = MADDPG_02(state_dim, action_dim, device)
            else:
                agents[agent_id] = SAC_02(state_dim, action_dim, device)

    # Training phase
    train(env, adversary_agents, cooperator_agents, 1000)
    # Evaluation phase
    mean_reward, std_reward = evaluate(env, adversary_agents, cooperator_agents, 1000)

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
        for type in agent_types:
            # Initialize empty array for storing the results
            results = []
            for algo in algorithms:
                individual = (agent_types == 'adversary')
                cooperative = (agent_types == 'cooperator')
                for env_name in env_names:
                    env, mean_reward, std_reward = run_experiment(env_name, algo, num_envs=args.num_envs)
                    results.append({'environment': env_name, 'algorithm': algo, 'setting': agent_types, 'source': source, 'mean_reward': mean_reward, 'std_reward': std_reward})
            # Saving the results to a CSV file for each setting and source
            filename = f'results/comparison_results_{source}_{agent_types}.csv'
            results_df = pd.DataFrame(results)
            results_df.to_csv(filename, index=False)
