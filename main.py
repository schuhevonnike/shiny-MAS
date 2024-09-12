import torch
import argparse
import numpy
from environments.pettingzoo_env import make_env
from algorithms01.dqn import DQNAgent
from utils.training import train
from utils.evaluation import evaluate

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def initialize_agents(env):
    if hasattr(env, 'unwrapped'):
        env = env.unwrapped
    agents = {}
    for agent_id in env.agent_iter:
        # Reshape observation and action space
        state_size = env.observation_space(agent_id).shape[0]
        action_size = env.action_space(agent_id).n
        # Create agents that inherit
        agents[agent_id] = DQNAgent(state_size, action_size)
    return agents

def run_experiment(env_fn, num_episodes):
    env = env_fn()
    # Initializing agents:
    print("Initializing Individual Agents...")
    individual_agents = initialize_agents(env)
    # Training agents:
    print("Training Individual Agents:")
    avg_rewards = train(individual_agents, num_episodes=num_episodes)
    # Print average individual agent results.
    print("\nAverage Rewards for Individual Agents:")
    for agent, reward in avg_rewards.items():
        print(f"{agent}: {reward:.2f}")
    env.close()
    # Evaluate agents
    print("Evaluating Individual Agents:")
    avg_rewards = evaluate(individual_agents, num_episodes=num_episodes)


if __name__ == "__main__":
    #print("Hello")
    parser = argparse.ArgumentParser(description="Multi-Agent Reinforcement Learning Comparison, to run in terminal print: py main.py (--algo) (--num_episodes)")
    #parser.add_argument('--algo', type=str, default='DQN', help='Algorithm to use: DQN, PPO, SAC, MADDPG')
    parser.add_argument('--num_episodes', type=int, default=16, help='Number of episodes for training each group of agents')
    args = parser.parse_args()
    run_experiment(env_fn=make_env, num_episodes=args.num_episodes)