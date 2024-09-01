#import torch
#import pandas
#import os
#import gymnasium
#from utils.training2 import train
#from utils.evaluation2 import evaluate


# Experiment parameters
#agent_types = ['adversary', 'cooperator']
# ...

# Cuda cores are better suited for such tasks, so ideally, we want to run the program on a cuda. If not possible, use the standard CPU.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#def run_experiment(env):
    # Initialize separate dictionaries for adversary and cooperator agents
    #adversary_agents = {}
    #cooperator_agents = {}

import argparse
from environments.pettingzoo_env import parallel_env
from algorithms01.dqn import DQNAgent
from algorithms01.ppo import PPOAgent
from algorithms01.sac import SACAgent
from algorithms01.maddpg import MADDPGAgent
from training2 import train
from evaluation2 import evaluate


def initialize_agents(env, algorithm, mode='individual'):
    agents = {}
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    for agent_id in env.possible_agents:
        if algorithm == 'DQN':
            agents[agent_id] = DQNAgent(state_size, action_size, cooperative=(mode == 'cooperative'))
        elif algorithm == 'PPO':
            agents[agent_id] = PPOAgent(state_size, action_size, cooperative=(mode == 'cooperative'))
        elif algorithm == 'SAC':
            agents[agent_id] = SACAgent(state_size, action_size, cooperative=(mode == 'cooperative'))
        elif algorithm == 'MADDPG':
            agents[agent_id] = MADDPGAgent(state_size, action_size, cooperative=(mode == 'cooperative'))
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    return agents


def run_experiment(algorithm, mode, num_episodes):
    env = parallel_env()

    # Initialize agents for individual and cooperative modes
    agents_individual = initialize_agents(env, algorithm, mode='individual')
    agents_cooperative = initialize_agents(env, algorithm, mode='cooperative')

    # Train agents
    print("Training Individual Agents:")
    rewards_individual = train(agents_individual, num_episodes=num_episodes, cooperative=False)

    print("Training Cooperative Agents:")
    rewards_cooperative = train(agents_cooperative, num_episodes=num_episodes, cooperative=True)

    # Evaluate agents
    print("Evaluating Individual Agents:")
    avg_rewards_individual = evaluate(agents_individual, num_episodes=num_episodes, cooperative=False)

    print("Evaluating Cooperative Agents:")
    avg_rewards_cooperative = evaluate(agents_cooperative, num_episodes=num_episodes, cooperative=True)

    # Compare results
    print(f"Average Rewards (Individual): {avg_rewards_individual}")
    print(f"Average Rewards (Cooperative): {avg_rewards_cooperative}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Reinforcement Learning Comparison")
    parser.add_argument('--algorithm', type=str, default='DQN', help='Algorithm to use: DQN, PPO, SAC, MADDPG')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes for training')

    args = parser.parse_args()

    run_experiment(algorithm=args.algorithm, mode='all', num_episodes=args.num_episodes)
