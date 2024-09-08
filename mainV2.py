import torch
import argparse
from environments.pettingzoo_env2 import make_env
from algorithms01.dqn import DQNAgent
from algorithms01.ppo import PPOAgent
from algorithms01.sac import SACAgent
from algorithms01.maddpg import MADDPGAgent
from utils.training2 import train
from utils.evaluation2 import evaluate

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def initialize_agents(env, algorithm, mode):
    if hasattr(env, 'unwrapped'):
        env = env.unwrapped

    agents = {}
    for agent_id in env.possible_agents:
        state_size = env.observation_space(agent_id).shape[0]
        action_size = env.action_space(agent_id).n

        if mode == 'cooperative':
            if algorithm == 'DQN':
                agents[agent_id] = DQNAgent(state_size, action_size, cooperative=True)
            elif algorithm == 'PPO':
                agents[agent_id] = PPOAgent(state_size, action_size, cooperative=True)
            elif algorithm == 'SAC':
                agents[agent_id] = SACAgent(state_size, action_size, cooperative=True)
            elif algorithm == 'MADDPG':
                agents[agent_id] = MADDPGAgent(state_size, action_size, cooperative=True)
        elif mode == 'individual':
            if algorithm == 'DQN':
                agents[agent_id] = DQNAgent(state_size, action_size, cooperative=False)
            elif algorithm == 'PPO':
                agents[agent_id] = PPOAgent(state_size, action_size, cooperative=False)
            elif algorithm == 'SAC':
                agents[agent_id] = SACAgent(state_size, action_size, cooperative=False)
            elif algorithm == 'MADDPG':
                agents[agent_id] = MADDPGAgent(state_size, action_size, cooperative=False)
    return agents


def run_experiment(env_fn, algorithm, num_episodes):
    env = env_fn()

    # Initialize agents
    print("Initializing Individual Agents...")
    agents_individual = initialize_agents(env, algorithm, mode='individual')
    print("Initializing Cooperative Agents...")
    agents_cooperative = initialize_agents(env, algorithm, mode='cooperative')

    # Train agents
    print("Training Individual Agents:")
    rewards_individual = train(agents_individual, num_episodes=num_episodes, cooperative=False)
    print("Training Cooperative Agents:")
    rewards_cooperative = train(agents_cooperative, num_episodes=num_episodes, cooperative=True)

    # Evaluate agents
    print("Evaluating Individual Agents:")
    rewards_history_individual, avg_rewards_individual, _ = evaluate(agents_individual, num_episodes=num_episodes,
                                                                     cooperative=False)

    print("Evaluating Cooperative Agents:")
    _, _, avg_rewards_cooperative = evaluate(agents_cooperative, num_episodes=num_episodes, cooperative=True)

    # Print individual agent results
    print("\nAverage Rewards for Individual Agents:")
    for agent, reward in avg_rewards_individual.items():
        print(f"{agent}: {reward:.2f}")

    # Print cooperative results
    print("\nCooperative Rewards:")
    print(f"Average Cooperative Reward: {avg_rewards_cooperative:.2f}")

    # Determine which mode performed better
    sum_avg_rewards_individual = sum(avg_rewards_individual.values()) if isinstance(avg_rewards_individual, dict) else 0
    sum_avg_rewards_cooperative = avg_rewards_cooperative if avg_rewards_cooperative is not None else 0

    print("\nComparison Result:")
    if sum_avg_rewards_individual > sum_avg_rewards_cooperative:
        print("Individual agents performed better.")
    elif sum_avg_rewards_individual < sum_avg_rewards_cooperative:
        print("Cooperative agents performed better.")
    else:
        print("Individual and cooperative agents performed equally well.")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Reinforcement Learning Comparison")
    parser.add_argument('--algorithm', type=str, default='DQN', help='Algorithm to use: DQN, PPO, SAC, MADDPG')
    parser.add_argument('--num_episodes', type=int, default=12, help='Number of episodes for training each group of agents')

    args = parser.parse_args()

    run_experiment(env_fn=make_env, algorithm=args.algorithm, num_episodes=args.num_episodes)
