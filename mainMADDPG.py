import torch
import argparse
from algorithms.maddpg import MADDPGAgent
from utils.trainvaluateMADDPG import train, evaluate
from utils.pettingzoo_env import make_env

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def initialize_agents(env):
    if hasattr(env, 'unwrapped'):
        env = env.unwrapped
    agents = {}
    for agent_id in env.possible_agents:
        # Reshape observation and action space.
        state_size = env.observation_space(agent_id).shape[0]
        action_size = env.action_space(agent_id).n
        # Create agents that inherit the attributes from the DQNAgent class.
        agents[agent_id] = MADDPGAgent(state_size, action_size)
    return agents

def run_experiment(env_fn, num_episodes):
    env = env_fn()
    # Initializing agents:
    print("\nInitializing Individual Agents...")
    cooperative_agents = initialize_agents(env)
    # Training agents:
    print("\nTraining Individual Agents:")
    avg_training_rewards = train(cooperative_agents, num_episodes=num_episodes)
    print(f"Type of avg_training_rewards in run_experiment: {type(avg_training_rewards)}") # Debugging
    # Print average individual agent training results:
    print("\nAverage Training Rewards for Individual Agents:")
    for agent, reward in avg_training_rewards.items():
        print(f"{agent}: {reward:.2f}")
    # Evaluate agents:
    print("\nEvaluating Individual Agents:")
    avg_evaluation_rewards = evaluate(cooperative_agents, num_episodes=num_episodes)
    # Print average individual agent evaluation results:
    print("\nAverage Evaluation Rewards for Individual Agents:")
    for agent, reward in avg_evaluation_rewards.items():
        print(f"{agent}: {reward:.2f}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Reinforcement Learning Comparison, to run in terminal print: py mainMADDPG.py (--num_episodes)")
    parser.add_argument('--num_episodes', type=int, default=12400, help='Number of episodes for training each group of agents')
    args = parser.parse_args()
    run_experiment(env_fn=make_env, num_episodes=args.num_episodes)