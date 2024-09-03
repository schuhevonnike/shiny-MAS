import torch
import argparse

from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.utils.conversions import parallel_to_aec, aec_to_parallel_wrapper
from environments.pettingzoo_env2 import make_env
from algorithms01.dqn import DQNAgent
from algorithms01.ppo import PPOAgent
from algorithms01.sac import SACAgent
from utils.training2 import train
from utils.evaluation2 import evaluate

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def initialize_agents(env, algorithm, mode='individual'):
    # Unwrap the environment if it is wrapped
    if hasattr(env, 'unwrapped'):
        env = env.unwrapped

    # Fetch the first agent from the list of possible agents to check the observation space
    first_agent = env.possible_agents[0] if len(env.possible_agents) > 0 else None
    print(f"Environment type: {type(env)}")
    print(f"First agent: {first_agent}")
    print(f"Observation space (first agent): {env.observation_space(first_agent)}")
    print(f"Action space (first agent): {env.action_space(first_agent)}")

    if first_agent is None:
        raise AttributeError("The environment does not have any possible agents.")

    # Get state and action sizes using the first agent
    if callable(env.observation_space):
        state_size = env.observation_space(first_agent).shape[0]
    else:
        state_size = env.observation_space.shape[0]

    if callable(env.action_space):
        action_size = env.action_space(first_agent).n
    else:
        action_size = env.action_space.n

    agents = {}
    for agent_id in env.possible_agents:
        cooperative = (mode == 'cooperative')  # Determine if agents should be cooperative
        if algorithm == 'DQN':
            agents[agent_id] = DQNAgent(state_size, action_size, cooperative=cooperative)
        elif algorithm == 'PPO':
            agents[agent_id] = PPOAgent(state_size, action_size, cooperative=cooperative)
        elif algorithm == 'SAC':
            agents[agent_id] = SACAgent(state_size, action_size, cooperative=cooperative)
        # Add other agent types as necessary

    return agents

def run_experiment(env_fn, algorithm, num_episodes=1000):
    # Initialize the environment
    env = env_fn()
    print(f"Original type of env: {type(env)}")

    # Convert environment to AEC if needed
    if isinstance(env, aec_to_parallel_wrapper):
        env = env.aec_env  # Access the underlying AEC environment
    elif isinstance(env, ParallelEnv):
        env = parallel_to_aec(env)  # Convert to AEC if necessary

    # Initialize agents for individual and cooperative modes
    print("Initializing Individual Agents...")
    agents_individual = initialize_agents(env, algorithm, mode='individual')
    print("Initializing Cooperative Agents...")
    agents_cooperative = initialize_agents(env, algorithm, mode='cooperative')

    # Train agents
    print("Training Individual Agents:")
    rewards_individual = train(agents_individual, env, num_episodes, cooperative=False)

    print("Training Cooperative Agents:")
    rewards_cooperative = train(agents_cooperative, env, num_episodes, cooperative=True)

    # Evaluate agents
    print("Evaluating Individual Agents:")
    avg_rewards_individual = evaluate(agents_individual, env, num_episodes, cooperative=False)

    print("Evaluating Cooperative Agents:")
    avg_rewards_cooperative = evaluate(agents_cooperative, env, num_episodes, cooperative=True)

    # Print and compare results
    print(f"Average Rewards (Individual): {avg_rewards_individual}")
    print(f"Average Rewards (Cooperative): {avg_rewards_cooperative}")

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

    run_experiment(env_fn=make_env(), algorithm=args.algorithm, num_episodes=args.num_episodes)
