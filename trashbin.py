#Code läuft bis zur Initialisierung des Environments, dann aber Problem mit der Observation
from pettingzoo.mpe import simple_tag_v3
from pettingzoo.utils import wrappers

def make_env():
    # Load the simple_tag_v3 environment as ParallelEnv
    env = simple_tag_v3.env()

    # Apply wrappers that are compatible with AEC environments
    env = wrappers.OrderEnforcingWrapper(env)
    return env

if __name__ == "__main__":
    env = make_env()

    # Initialize the environment and get initial observations
    env.reset()
    print(f"Environment initialized")  # Debugging print to verify output

    #agent_iter() is a method used to iterate over all possible agents
    for agent in env.agent_iter():
        #Retrieve obs, rew and further info from the previous agent/step
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample()  # Take random actions

        # Execute the action
        env.step(action)  # Do not unpack, as step does not return anything

        if termination or truncation:
            env.reset()  # Reset the environment for the next episode

    env.render()  # Adjust render mode as needed
    env.close()

#08.09.24 - old evaluation2.py
import torch
import pandas as pd
from environments.pettingzoo_env2 import make_env

def evaluate(agents, num_episodes=10, cooperative=False):
    env = make_env()
    rewards_history = {agent: [] for agent in agents.keys()} # Dict for individual rewards
    total_cooperative_rewards = [] # List for tracking cooperative rewards

    for episode in range(num_episodes):
        env.reset(seed=42)
        total_rewards = {agent: 0 for agent in env.possible_agents}
        done = False  # Flag to keep track of and manage episode completion

        while not done:
            for agent in env.agent_iter():
                observation, reward, termination, truncation, _ = env.last()

                if termination or truncation:
                    env.step(None)  # Step with None to signify no action taken for ended agent
                    done = True  # End the episode if any agent is done
                    break  # Exit agent iteration

                obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    if cooperative:
                        action = agents[agent].act(obs_tensor)
                    else:
                        action = torch.argmax(agents[agent].model(obs_tensor)).item()

                env.step(action)

                # Update total rewards for this agent
                if cooperative:
                    # In cooperative mode, just aggregate the rewards from all agents
                    total_rewards[agent] += reward  # Keep track of individual rewards
                else:
                    total_rewards[agent] += reward  # Individual rewards

            if cooperative:
                # For cooperative mode, calculate the total shared reward
                total_shared_reward = sum(total_rewards.values())
                total_cooperative_rewards.append(total_shared_reward)
                print(f"Episode {episode + 1}/{num_episodes} | Cooperative Total Reward: {total_shared_reward}")
            else:
                # For non-cooperative mode, store individual rewards
                for agent in total_rewards:
                    rewards_history[agent].append(total_rewards[agent])
                print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")

        env.close()

        # Calculate average rewards for each agent in individual mode
        if not cooperative:
            avg_rewards_individual = {agent: sum(rewards) / len(rewards) for agent, rewards in
                                          rewards_history.items()}
            avg_rewards_cooperative = None  # No cooperative mode here
        else:
            avg_rewards_individual = None  # No individual mode in cooperative case
            avg_rewards_cooperative = sum(total_cooperative_rewards) / len(total_cooperative_rewards)

        return rewards_history, avg_rewards_individual, avg_rewards_cooperative

#Old evaluation2.py
def evaluate(agents, num_episodes=10, cooperative=False):
    env = make_env()
    rewards_history = {agent: [] for agent in agents.keys()}

    for episode in range(num_episodes):
        env.reset()
        total_rewards = {agent: 0 for agent in env.possible_agents}

        for agent in env.agent_iter():
            observation, reward, termination, truncation, _ = env.last()

            if termination or truncation:
                env.step(None)  # Step with None to signify no action taken for ended agent
                continue

            # Select action based on the mode (cooperative or individual)
            with torch.no_grad():
                # "Replace torch.tensor(..., dtype=...) with target = target.clone().detach().float().unsqueeze(0)"
                # Convert float to PyTorch tensor
                obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                # Use tensor operations
                obs_tensor = obs_tensor.clone().detach().float().unsqueeze(0)
                if cooperative:
                    # Cooperative mode: choose action considering cooperative strategy
                    action = agents[agent].act(obs_tensor)
                else:
                    # Individual mode: choose action independently
                    action = torch.argmax(agents[agent].model(obs_tensor)).item()

            env.step(action)

            # Update total rewards for this agent
            if cooperative:
                shared_reward = sum(reward.values())
                for a in total_rewards:
                    total_rewards[a] += shared_reward
            else:
                total_rewards[agent] += reward

        # Store total rewards per agent after each episode
        for agent in total_rewards:
            rewards_history[agent].append(total_rewards[agent])

        print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")

    env.close()

    # Calculate average rewards for each agent
    avg_rewards = {agent: sum(rewards) / len(rewards) for agent, rewards in rewards_history.items()}
    return avg_rewards

#08.09.24 - Old evaluation2.py:

import torch
from environments.pettingzoo_env2 import make_env

def evaluate(agents, num_episodes=10, cooperative=False):
    env = make_env()
    rewards_history = {agent: [] for agent in agents.keys()}  # Dict for individual rewards
    total_cooperative_rewards = []  # List for tracking cooperative rewards

    for episode in range(num_episodes):
        env.reset(seed=42)
        total_rewards = {agent: 0 for agent in env.possible_agents}
        done = False  # Flag to keep track of and manage episode completion

        while not done:
            for agent in env.agent_iter():
                observation, reward, termination, truncation, _ = env.last()

                if termination or truncation:
                    env.step(None)  # Step with None to signify no action taken for ended agent
                    done = True  # End the episode if any agent is done
                    break  # Exit agent iteration

                obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    if cooperative:
                        action = agents[agent].act(obs_tensor)
                    else:
                        action = torch.argmax(agents[agent].model(obs_tensor)).item()

                env.step(action)

                # Update total rewards for this agent
                total_rewards[agent] += reward  # Keep track of individual rewards

            # Check if cooperative mode
            if cooperative:
                # Calculate the total shared reward across all agents
                shared_reward = sum(total_rewards.values())
                total_cooperative_rewards.append(shared_reward)
                print(f"Episode {episode + 1}/{num_episodes} | Cooperative Total Reward: {shared_reward}")
            else:
                # For non-cooperative mode, store individual rewards
                for agent in total_rewards:
                    rewards_history[agent].append(total_rewards[agent])
                print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")

        env.close()

    # Calculate average rewards for each agent in individual mode
    if not cooperative:
        # Average rewards per agent
        avg_rewards_individual = {agent: sum(rewards) / len(rewards) for agent, rewards in rewards_history.items()}
        avg_rewards_cooperative = None  # No cooperative mode average reward
    else:
        # Average cooperative rewards
        avg_rewards_individual = None  # No individual mode average reward
        avg_rewards_cooperative = sum(total_cooperative_rewards) / len(total_cooperative_rewards)

    return rewards_history, avg_rewards_individual, avg_rewards_cooperative

#06.09.24 - Old training2.py:
import torch
from environments.pettingzoo_env2 import make_env
from algorithms01.dqn import DQNAgent
from algorithms01.maddpg import MADDPGAgent
from algorithms01.ppo import PPOAgent
from algorithms01.sac import SACAgent


def select_action(agent, observation, cooperative=False, other_agents=None):
    with torch.no_grad():
        if isinstance(agent, DQNAgent):
            action = agent.act(observation)
        elif isinstance(agent, PPOAgent):
            action, _ = agent.act(observation)
        elif isinstance(agent, SACAgent):
            action, _ = agent.act(observation)
        elif isinstance(agent, MADDPGAgent):
            action = agent.act(observation)
        else:
            raise ValueError("Unknown agent type")
    return action


def train(agents, num_episodes=10, cooperative=False):
    env = make_env()
    # Check if `env` is an environment instance and not a wrapper object
    if not hasattr(env, 'reset') or not hasattr(env, 'step'):
        raise TypeError("Provided env is not a valid environment instance")

    rewards_history = {agent: [] for agent in agents}
    gamma = 0.99  # Discount factor, adjustable based on algorithm
    batch_size = 64  # Batch size, relevant for algorithms with replay memory

    for episode in range(num_episodes):
        env.reset()
        total_rewards = {agent: 0 for agent in env.possible_agents}

        for agent in env.agent_iter():
            observation, reward, termination, truncation, _ = env.last()

            if termination or truncation:
                env.step(None)  # Tell the environment no action will be taken
                continue

            # Select action based on the mode (cooperative or individual)
            action = select_action(agents[agent], observation, cooperative, other_agents=agents.values())
            env.step(action)

            next_observation, reward, termination, truncation, _ = env.last()
            total_rewards[agent] += reward

            # Store experience and update agent model
            agents[agent].remember(observation, action, reward, next_observation, termination or truncation)

            if isinstance(agents[agent], DQNAgent):
                if len(agents[agent].memory) >= batch_size:
                    agents[agent].replay(batch_size)
            elif isinstance(agents[agent], MADDPGAgent):
                if len(agents[agent].memory) >= batch_size:
                    agents[agent].update(batch_size)
            elif isinstance(agents[agent], PPOAgent):
                if len(agents[agent].memory) >= batch_size:
                    agents[agent].update()
            elif isinstance(agents[agent], SACAgent):
                if len(agents[agent].memory) >= batch_size:
                    agents[agent].update(batch_size)

        # Log rewards
        for agent in total_rewards:
            rewards_history[agent].append(total_rewards[agent])

        print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")

    env.close()
    return rewards_history

#Another one

def train(agents, num_episodes=10, cooperative=False):
    env = make_env()
    # Check if `env` is an environment instance and not a wrapper object
    if not hasattr(env, 'reset') or not hasattr(env, 'step'):
        raise TypeError("Provided env is not a valid environment instance")

    rewards_history = {agent: [] for agent in agents}
    gamma = 0.99  # Discount factor, adjustable based on algorithm
    batch_size = 64  # Batch size, relevant for algorithms with replay memory

    for episode in range(num_episodes):
        env.reset()
        total_rewards = {agent: 0 for agent in env.possible_agents}
        print(f"Episode {episode + 1}/{num_episodes} started.")

        done = False  # Flag to manage episode completion

        while not done:
            for agent in env.agent_iter():
                observation, reward, termination, truncation, _ = env.last()
                print(f"Agent: {agent}, Observation: {observation}, Reward: {reward}")

                if termination or truncation:
                    done = True
                    env.step(None)  # Tell the environment no action will be taken
                    break  # Exit agent iteration if episode ended

                # Select action based on the mode (cooperative or individual)
                action = select_action(agents[agent], observation, cooperative, other_agents=agents.values())
                print(f"Selected Action for {agent}: {action}")

                env.step(action)

                next_observation, reward, termination, truncation, _ = env.last()
                total_rewards[agent] += reward

                # Store experience and update agent model
                agents[agent].remember(observation, action, reward, next_observation, termination or truncation)

                # Perform agent-specific updates
                if isinstance(agents[agent], DQNAgent):
                    if len(agents[agent].memory) >= batch_size:
                        agents[agent].replay(batch_size)
                elif isinstance(agents[agent], MADDPGAgent):
                    if len(agents[agent].memory) >= batch_size:
                        agents[agent].update(batch_size)
                elif isinstance(agents[agent], PPOAgent):
                    if len(agents[agent].memory) >= batch_size:
                        agents[agent].update()
                elif isinstance(agents[agent], SACAgent):
                    if len(agents[agent].memory) >= batch_size:
                        agents[agent].update(batch_size)

        # Log rewards
        for agent in total_rewards:
            rewards_history[agent].append(total_rewards[agent])
        print(f"Episode {episode + 1} completed with Total Rewards: {total_rewards}")

    env.close()
    return rewards_history

import time
import torch
import pandas as pd
from environments.pettingzoo_env2 import make_env
from algorithms01.dqn import DQNAgent
from algorithms01.maddpg import MADDPGAgent
from algorithms01.ppo import PPOAgent
from algorithms01.sac import SACAgent

def select_action(agent, observation, cooperative=False, other_agents=None):
    # Tipp: Diese Instanzen sind unnötig, besser ist es, diese Checks bei den einzelnen Lernalgorithmen mit einzubinden
    with torch.no_grad():
        if isinstance(agent, DQNAgent):
            action = agent.act(observation)
        elif isinstance(agent, PPOAgent):
            action, _ = agent.act(observation)
        elif isinstance(agent, SACAgent):
            action, _ = agent.act(observation)
        elif isinstance(agent, MADDPGAgent):
            action = agent.act(observation)
        else:
            raise ValueError("Unknown agent type")
    return action

def train(agents, num_episodes=10, cooperative=False):
    env = make_env()
    rewards_history = {agent: [] for agent in agents}
    data_records = {agent: {'episode': [], 'observation': [], 'action': [], 'reward': [], 'next_observation': [], 'done': [],'step_duration': []} for agent in agents}
    gamma = 0.99  # Discount factor, adjustable based on algorithm
    batch_size = 64  # Batch size, relevant for algorithms with replay memory
    # Check if `env` is an environment instance and not a wrapper object
    if not hasattr(env, 'reset') or not hasattr(env, 'step'):
        raise TypeError("Provided env is not a valid environment instance")

    for episode in range(num_episodes):
        episode_start_time = time.time()
        env.reset() # Returns NoneType
        total_rewards = {agent: 0 for agent in env.possible_agents}
        print(f"Episode {episode + 1}/{num_episodes} started.")

        done = False  # Flag to manage episode completion
        try:
            while not done:
                # For loop over all possible Agents
                for agent in env.agent_iter():
                    step_start_time = time.time()
                    observation, reward, termination, truncation, _ = env.last()
                    print(f"Agent: {agent}, Observation: {observation}, Reward: {reward}")

                    if termination or truncation:
                        done = True
                        env.step(None)  # Tell the environment no action will be taken
                        break  # Exit agent iteration if episode ended

                    # Select action based on the mode (cooperative or individual)
                    action = select_action(agents[agent], observation, cooperative=False, other_agents=agents.values())
                    print(f"Selected Action for {agent}: {action}")
                    #data[agent].append
                    env.step(action)

                    next_observation, reward, termination, truncation, _ = env.last()

                    total_rewards[agent] += reward

                    # Timing the agent's memory and update operations
                    agent_start_time = time.time()

                    # Store data in the dictionary
                    data_records[agent]['episode'].append(episode + 1)
                    data_records[agent]['observation'].append(observation)
                    data_records[agent]['action'].append(action)
                    data_records[agent]['reward'].append(reward)
                    data_records[agent]['next_observation'].append(next_observation)
                    data_records[agent]['done'].append(termination or truncation)
                    data_records[agent]['step_duration'].append(time.time() - step_start_time)

                    # Store experience and update agent model
                    agents[agent].remember(observation, action, reward, next_observation, termination or truncation)

                    # Perform agent-specific updates => sollte in die einzelnen Lernalgorithmen implementiert werden
                    if isinstance(agents[agent], DQNAgent):
                        if len(agents[agent].memory) >= batch_size:
                            agents[agent].replay(batch_size)
                    elif isinstance(agents[agent], MADDPGAgent):
                        if len(agents[agent].memory) >= batch_size:
                            agents[agent].update(batch_size)
                    elif isinstance(agents[agent], PPOAgent):
                        if len(agents[agent].memory) >= batch_size:
                            agents[agent].update()
                    elif isinstance(agents[agent], SACAgent):
                        if len(agents[agent].memory) >= batch_size:
                            agents[agent].update(batch_size)
                        # Time taken for agent operations
                        print(f"Agent {agent} operation took {time.time() - agent_start_time:.2f} seconds")
                    # Time taken for a single environment step
                    print(f"Step operation took {time.time() - step_start_time:.2f} seconds")
                # Timing the end of an episode
                print(f"Episode {episode + 1} completed in {time.time() - episode_start_time:.2f} seconds with Total Rewards: {total_rewards}")
                # Log rewards
            for agent in total_rewards:
                rewards_history[agent].append(total_rewards[agent])

        except KeyboardInterrupt:
            print("Training interrupted by user.")
        finally:
            print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")
            env.close()

    # Convert data_records to pandas DataFrame
    all_agent_data = []
    for agent, data in data_records.items():
        df_agent = pd.DataFrame(data)
        df_agent['agent'] = agent  # Add a column for the agent
        all_agent_data.append(df_agent)

    df_eval = pd.concat(all_agent_data, ignore_index=True)
    df_eval.to_csv('evaluation_data.csv', index=False)  # Save evaluation data to CSV for analysis

    # Calculate average rewards for each agent
    #!!!BUGGY!!!
    avg_rewards = {agent: sum(rewards) / len(rewards) for agent, rewards in rewards_history.items()}
    return avg_rewards
    #return rewards_history

#08.09.24 - old training2.py
import time
import torch
import pandas as pd
from torch.distributions.utils import logits_to_probs

from environments.pettingzoo_env2 import make_env
from algorithms01.dqn import DQNAgent
from algorithms01.maddpg import MADDPGAgent
from algorithms01.ppo import PPOAgent
from algorithms01.sac import SACAgent

def select_action(agent, observation, cooperative=False, other_agents=None):
    with torch.no_grad():
        if isinstance(agent, DQNAgent):
            action = agent.act(observation)
        elif isinstance(agent, PPOAgent):
            action, _ = agent.act(observation)
        elif isinstance(agent, SACAgent):
            action, _ = agent.act(observation)
        elif isinstance(agent, MADDPGAgent):
            action = agent.act(observation)
        else:
            raise ValueError("Unknown agent type")
    return action

def train(agents, num_episodes=10, cooperative=False):
    env = make_env()
    rewards_history = {agent: [] for agent in agents}
    data_records = {agent: {'episode': [], 'observation': [], 'action': [], 'reward': [], 'next_observation': [], 'done': [], 'step_duration': []} for agent in agents}
    gamma = 0.99  # Discount factor
    batch_size = 64  # Batch size for algorithms with replay memory
    total_cooperative_rewards = []  # Track cooperative rewards

    if not hasattr(env, 'reset') or not hasattr(env, 'step'):
        raise TypeError("Provided env is not a valid environment instance")

    for episode in range(num_episodes):
        episode_start_time = time.time()
        env.reset()
        total_rewards = {agent: 0 for agent in env.possible_agents}
        print(f"Episode {episode + 1}/{num_episodes} started.")
        done = False

        try:
            while not done:
                for agent in env.possible_agents:
                    step_start_time = time.time()
                    observation, reward, termination, truncation, _ = env.last()
                    print(f"Agent: {agent}, Observation: {observation}, Reward: {reward}")

                    if termination or truncation:
                        done = True
                        env.step(None)
                        break

                    # Select action based on the mode
                    action = select_action(agents[agent], observation, cooperative=cooperative, other_agents=agents.values())
                    print(f"Selected Action for {agent}: {action}")
                    env.step(action)

                    next_observation, reward, termination, truncation, _ = env.last()

                    # Store experience and update agent model
                    agents[agent].remember(observation, action, reward, next_observation, termination or truncation)

                    # Update agent
                    if len(agents[agent].memory) >= batch_size:
                        agents[agent].update(batch_size)

                    # Store data in the dictionary
                    data_records[agent]['episode'].append(episode + 1)
                    data_records[agent]['observation'].append(observation)
                    data_records[agent]['action'].append(action)
                    data_records[agent]['reward'].append(reward)
                    data_records[agent]['next_observation'].append(next_observation)
                    data_records[agent]['done'].append(termination or truncation)
                    data_records[agent]['step_duration'].append(time.time() - step_start_time)

                    # Update total rewards
                    total_rewards[agent] += reward

                    print(f"Step operation took {time.time() - step_start_time:.2f} seconds")

            # In cooperative mode, aggregate total rewards for all agents
            if cooperative:
                # Calculate the shared reward based on the rewards history
                shared_reward = sum([sum(rewards_history[agent]) for agent in rewards_history])
                total_cooperative_rewards.append(shared_reward)
                print(f"Episode {episode + 1}/{num_episodes} | Cooperative Total Reward: {shared_reward}")
            else:
                # Append the individual rewards for each agent to the rewards history
                for agent in total_rewards:
                    rewards_history[agent].append(total_rewards[agent])
                print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")

        except KeyboardInterrupt:
            print("Training interrupted by user.")
        finally:
            print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")
            env.close()

    # Convert data_records to pandas DataFrame
    all_agent_data = []
    for agent, data in data_records.items():
        df_agent = pd.DataFrame(data)
        df_agent['agent'] = agent  # Add a column for the agent
        all_agent_data.append(df_agent)

    df_eval = pd.concat(all_agent_data, ignore_index=True)
    df_eval.to_csv('evaluation_data.csv', index=False)  # Save evaluation data to CSV for analysis

    # Calculate average rewards for each agent or cooperative rewards
    if cooperative:
        # Average cooperative rewards based on total_cooperative_rewards
        avg_rewards = sum(total_cooperative_rewards) / len(total_cooperative_rewards)
    else:
        # Average individual rewards based on the rewards_history
        avg_rewards = {agent: sum(rewards) / len(rewards) for agent, rewards in rewards_history.items()}

    return avg_rewards


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
    sum_avg_rewards_individual = sum(avg_rewards_individual)
    sum_avg_rewards_cooperative = sum(avg_rewards_cooperative)

    # Now compare the summed average rewards
    if sum_avg_rewards_individual > sum_avg_rewards_cooperative:
        print("Individual agents performed better.")
    elif sum_avg_rewards_individual < sum_avg_rewards_cooperative:
        print("Cooperative agents performed better.")
    else:
        print("Individual and cooperative agents performed equally well.")

    env.close()

#08.09.24