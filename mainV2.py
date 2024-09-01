import torch
import argparse

from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.utils.conversions import parallel_to_aec, aec_to_parallel, aec_to_parallel_wrapper
from pettingzoo.utils.wrappers import OrderEnforcingWrapper, TerminateIllegalWrapper

from environments.pettingzoo_env2 import parallel_env
from algorithms01.dqn import DQNAgent
from algorithms01.ppo import PPOAgent
from algorithms01.sac import SACAgent
from algorithms01.maddpg import MADDPGAgent
from utils.training2 import train
from utils.evaluation2 import evaluate

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def initialize_agents(env, algorithm, mode='individual'):
    # Unwrap the environment if it is wrapped
    if hasattr(env, 'unwrapped'):
        env = env.unwrapped

    # Determine the correct environment format
    if hasattr(env, 'observation_space') and hasattr(env.observation_space, 'shape'):
        state_size = env.observation_space.shape[0]
    else:
        raise AttributeError("The environment does not have a valid observation_space.")

    action_size = env.action_space.n

    agents = {}
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

def run_experiment(env_fn, algorithm, num_episodes):
    # Initialize the environment
    env = env_fn()
    print(f"Original type of env: {type(env)}")

    # Convert environment to AEC if needed
    if isinstance(env, aec_to_parallel_wrapper):
        env = env.aec_env  # Access the underlying AEC environment
    elif isinstance(env, ParallelEnv):
        env = parallel_to_aec(env)  # Convert to AEC if necessary

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

    # Print and use results
    print(f"Average Rewards (Individual): {avg_rewards_individual}")
    print(f"Average Rewards (Cooperative): {avg_rewards_cooperative}")

    # Compare results if needed
    if avg_rewards_individual > avg_rewards_cooperative:
        print("Individual agents performed better.")
    elif avg_rewards_individual < avg_rewards_cooperative:
        print("Cooperative agents performed better.")
    else:
        print("Individual and cooperative agents performed equally well.")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Reinforcement Learning Comparison")
    parser.add_argument('--algorithm', type=str, default='DQN', help='Algorithm to use: DQN, PPO, SAC, MADDPG')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes for training')

    args = parser.parse_args()

    run_experiment(env_fn=parallel_env, algorithm=args.algorithm, num_episodes=args.num_episodes)
