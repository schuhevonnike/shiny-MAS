import torch
import argparse
import numpy
from environments.pettingzoo_env import make_env
from algorithms01.dqn import DQNAgent
from utils.training import train
from utils.evaluation import evaluate

device = 'cuda'


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def initialize_agents(env):
    agents = {}
    for agent_id in env.possible_agents:
        # Reshape observation and action space
        state_size = env.observation_space(agent_id).shape[0]
        action_size = env.action_space(agent_id).n
        # Create agents that inherit
        agents[agent_id] = DQNAgent(state_size, action_size)
    return agents

def run_experiment(env_fn, algo, num_episodes):
    env = env_fn()
    # Initialize agents
    print("Initializing Individual Agents...")
    individual_agents = initialize_agents(env)
    # Train agents
    print("Training Individual Agents:")
    individual_rewards = train(individual_agents, num_episodes=num_episodes)
    # Evaluate agents
    print("Evaluating Individual Agents:")
    rewards_history, avg_individual_rewards = evaluate(individual_agents, num_episodes=num_episodes)
    # Print individual agent results
    print("\nAverage Rewards for Individual Agents:")
    for agent, reward in avg_individual_rewards.items():
        print(f"{agent}: {reward:.2f}")
    env.close()

if __name__ == "__main__":
    #print("Hello")
    parser = argparse.ArgumentParser(description="Multi-Agent Reinforcement Learning Comparison, to run in terminal print: py main.py (--algo) (--num_episodes)")
    parser.add_argument('--algo', type=str, default='DQN', help='Algorithm to use: DQN, PPO, SAC, MADDPG')
    parser.add_argument('--num_episodes', type=int, default=16, help='Number of episodes for training each group of agents')
    args = parser.parse_args()
    run_experiment(env_fn=make_env, algo=args.algo, num_episodes=args.num_episodes)