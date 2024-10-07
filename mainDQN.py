import os

import torch
import argparse

from algorithms.dqn import DQNAgent
from utils.trainvaluateDQN import train, evaluate
from utils.pettingzoo_env import make_env

NUM_THREADS = 1
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(NUM_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_THREADS)
torch.set_num_threads(NUM_THREADS)

#device = 'cuda' if torch.cuda.is_available() else 'cpu'

def initialize_agents(env):
    if hasattr(env, 'unwrapped'):
        env = env.unwrapped
    agents = {}
    for agent_id in env.possible_agents:
        # Reshape observation and action space.
        state_size = env.observation_space(agent_id).shape[0]
        action_size = env.action_space(agent_id).n
        # Create agents that inherit the attributes from the DQNAgent class.
        agents[agent_id] = DQNAgent(state_size, action_size)
    return agents

def run_experiment(env_fn, num_episodes,seed):
    env = env_fn()
    # Initializing agents:
    print("\nInitializing Individual Agents...")
    agents = initialize_agents(env)
    # Training agents:
    print("\nTraining Individual Agents:")
    avg_training_rewards = train(agents, num_episodes=num_episodes, seed=seed)
    # Print average individual agent training results:
    print("\nAverage Training Rewards for Individual Agents:")
    for agent, reward in avg_training_rewards.items():
        print(f"{agent}: {reward:.2f}")
    # Evaluate agents:
    print("\nEvaluating Individual Agents:")
    avg_evaluation_rewards = evaluate(agents, num_episodes=num_episodes, seed=seed)
    # Print average individual agent evaluation results:
    print("\nAverage Evaluation Rewards for Individual Agents:")
    for agent, reward in avg_evaluation_rewards.items():
        print(f"{agent}: {reward:.2f}")
    env.close()

def start_seed(seed):
    parser = argparse.ArgumentParser(description="Multi-Agent Reinforcement Learning Comparison, to run in terminal print: py mainDQN.py (--num_episodes)")
    parser.add_argument('--num_episodes', type=int, default=12400, help='Number of episodes for training each group of agents')
    args = parser.parse_args()
    run_experiment(env_fn=make_env, num_episodes=args.num_episodes, seed=seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes for training each group of agents')
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    run_experiment(env_fn=make_env, num_episodes=args.num_episodes, seed=args.seed)
