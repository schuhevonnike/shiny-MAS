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
    # Unwrap the environment if it is wrapped
    if hasattr(env, 'unwrapped'):
        env = env.unwrapped

    # Fetch the first agent from the list of possible agents to check the observation space
    #first_agent = env.possible_agents[0] if len(env.possible_agents) > 0 else None
    #print(f"Environment type: {type(env)}")
    #print(f"First agent: {first_agent}")
    # Check the observation and action space for the first agent
    #obs_space = env.observation_space(first_agent) if callable(env.observation_space) else env.observation_space
    #act_space = env.action_space(first_agent) if callable(env.action_space) else env.action_space
    #print(f"Observation space (first agent): {obs_space}")
    #print(f"Observation shape (first agent): {obs_space.shape}")
    #print(f"Action space (first agent): {act_space}")
    #print(f"Action shape (first agent): {act_space.shape}")

    #if first_agent is None:
    #    raise AttributeError("The environment does not have any possible agents.")

    # Get state and action sizes
    #state_size = obs_space.shape[0] if hasattr(obs_space, 'shape') else obs_space.n
    #action_size = act_space.n if hasattr(act_space, 'n') else act_space.shape[0]

    agents = {}

    for agent_id in env.possible_agents:
        # Dynamic gathering of observation and action space size for each agent
        state_size = env.observation_space(agent_id).shape[0]
        print(f"State size inferred: {state_size}")
        action_size = env.action_space(agent_id).n
        print(f"Action size inferred: {action_size}")

        if mode == 'cooperative':
            #cooperative = (mode == 'cooperative')  # Introduce new boolean to annotate agents behaviour
            if algorithm == 'DQN':
                agents[agent_id] = DQNAgent(state_size, action_size, mode)
                print(f"Initialized cooperative DQNAgent with state size: {state_size}")
            elif algorithm == 'PPO':
                agents[agent_id] = PPOAgent(state_size, action_size, mode)
                print(f"Initialized cooperative PPOAgent with state size: {state_size}")
            elif algorithm == 'SAC':
                agents[agent_id] = SACAgent(state_size, action_size, mode)
                print(f"Initialized cooperative SACAgent with state size: {state_size}")
            elif algorithm == 'MADDPG':
                agents[agent_id] = MADDPGAgent(state_size, action_size, mode)
                print(f"Initialized cooperative SACAgent with state size: {state_size}")
        elif mode == 'individual':
            if algorithm == 'DQN':
                agents[agent_id] = DQNAgent(state_size, action_size, mode)
                print(f"Initialized individual DQNAgent with state size: {state_size}")
            elif algorithm == 'PPO':
                agents[agent_id] = PPOAgent(state_size, action_size, mode)
                print(f"Initialized individual PPOAgent with state size: {state_size}")
            elif algorithm == 'SAC':
                agents[agent_id] = SACAgent(state_size, action_size, mode)
                print(f"Initialized individual SACAgent with state size: {state_size}")
            elif algorithm == 'MADDPG':
                agents[agent_id] = MADDPGAgent(state_size, action_size, mode)
                print(f"Initialized individual SACAgent with state size: {state_size}")
    return agents

def run_experiment(env_fn, algorithm, num_episodes):
    # Initialize the environment
    env = env_fn()
    print(f"Original type of env: {type(env)}")

    # Initialize agents for individual and cooperative modes
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
    avg_rewards_individual = evaluate(agents_individual, num_episodes=num_episodes, cooperative=False)

    print("Evaluating Cooperative Agents:")
    avg_rewards_cooperative = evaluate(agents_cooperative, num_episodes=num_episodes, cooperative=True)

    # Print and compare results
    print(f"Average Rewards (Individual): {avg_rewards_individual}")
    print(f"Average Rewards (Cooperative): {avg_rewards_cooperative}")

    # Calculate the sum of average rewards for individual and cooperative agents
    sum_avg_rewards_individual = sum(avg_rewards_individual.values())
    sum_avg_rewards_cooperative = sum(avg_rewards_cooperative.values())

    # Now compare the summed average rewards
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
    parser.add_argument('--num_episodes', type=int, default=4, help='Number of episodes for training each group of agents')

    args = parser.parse_args()

    run_experiment(env_fn=make_env, algorithm=args.algorithm, num_episodes=args.num_episodes)
