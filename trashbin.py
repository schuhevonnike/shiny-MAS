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

#09.09.24

def evaluate(agents, num_episodes=10, cooperative=False):
    env = make_env()
    rewards_history = {agent: [] for agent in agents.keys()}  # Individual rewards
    total_cooperative_rewards = []  # Cooperative rewards
    data_records = []  # List for step-by-step evaluation data

    for episode in range(num_episodes):
        #episode_start_time = time.time()
        env.reset(seed=42)
        total_rewards = {agent: 0 for agent in env.possible_agents}
        done = False
        #step = 0

        while not done:
            #step += 1
            #for agent in env.agent_iter():
            for agent in env.possible_agents:
                observation, reward, termination, truncation, _ = env.last()

                if termination or truncation:
                    env.step(None)  # No action taken for terminated agents
                    done = True
                    break

                obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    if cooperative:
                        action = agents[agent].act(obs_tensor)
                    else:
                        action = torch.argmax(agents[agent].model(obs_tensor)).item()

                env.step(action)

                # Update rewards
                total_rewards[agent] += reward

                next_observation, reward, termination, truncation, _ = env.last()

                # Store step data
                data_records.append({
                    'Episode': episode + 1,
                    #'Step': step,
                    'Agent': agent,
                    'Mode': 'cooperative' if cooperative else 'individual',
                    'Action': action,
                    'Observation': observation,
                    'Next Observation': next_observation,
                    'Reward': reward,
                    'Total Reward': total_rewards[agent],
                    'Cooperative Reward': sum(total_rewards.values()) if cooperative else 0,
                    'Done': termination or truncation,
                    #'Step Duration': time.time() - episode_start_time
                })

            if cooperative:
                # Track cooperative rewards
                total_shared_reward = sum(total_rewards.values())
                total_cooperative_rewards.append(total_shared_reward)
                print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")
                #print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_shared_reward}")
            else:
                # Track individual rewards
                for agent in total_rewards:
                    rewards_history[agent].append(total_rewards[agent])
                    print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")

        # Convert records to a DataFrame and save to CSV
        df_eval = pd.DataFrame(data_records)

        # Ensure directory exists for saving
        if not os.path.exists('evaluation_data'):
            os.makedirs('evaluation_data')

        # Save evaluation data to CSV
        df_eval.to_csv('evaluation_data/evaluation_data.csv', index=False)
        print(f"Evaluation data saved to evaluation_data/evaluation_data.csv")
    env.close()

    # Calculate average rewards
    if not cooperative:
        avg_rewards_individual = {agent: sum(rewards) / len(rewards) for agent, rewards in rewards_history.items()}
        avg_rewards_cooperative = None
    else:
        avg_rewards_individual = None
        avg_rewards_cooperative = sum(total_cooperative_rewards) / len(total_cooperative_rewards)

    return rewards_history, avg_rewards_individual, avg_rewards_cooperative

#09.09.24 - working fine, yet training rewards are off for advesary_2.

import os
#import time
import torch
import pandas as pd
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
    data_records = []  # List to store records per step
    total_cooperative_rewards = []  # Track cooperative rewards

    if not hasattr(env, 'reset') or not hasattr(env, 'step'):
        raise TypeError("Provided env is not a valid environment instance")

    for episode in range(num_episodes):
        #episode_start_time = time.time()
        env.reset(seed=42)
        total_rewards = {agent: 0 for agent in env.possible_agents}
        done = False
        step = 0  # Initialize step counter for each episode

        while not done:
            step += 1
            for agent in env.possible_agents:
            #for agent in env.agent_iter():
                observation, reward, termination, truncation, _ = env.last()

                #obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

                action = select_action(agents[agent], observation, cooperative=cooperative, other_agents=agents.values())

                env.step(action)
                next_observation, reward, termination, truncation, _ = env.last()

                # Update rewards
                total_rewards[agent] += reward

                if termination or truncation:
                    done = True
                    env.step(None)
                    break

                agents[agent].remember(observation, action, reward, next_observation, termination or truncation)

                # Store step data in the list
                data_records.append({
                    'Episode': episode + 1,
                    'Step': step,
                    'Agent': agent,
                    'Mode': 'cooperative' if cooperative else 'individual',
                    'Action': action,
                    'Observation': observation,
                    'Next Observation': next_observation,
                    'Reward': reward,
                    'Total Reward': total_rewards[agent],
                    'Cooperative Reward': sum(total_rewards.values()) if cooperative else 0,
                    'Done': termination or truncation,
                    #'Step Duration': time.time() - episode_start_time
                })

                # Update agent if it has enough experience (for agents with replay memory)
                if len(agents[agent].memory) >= 64:  # Using batch size 64 for example
                    agents[agent].update(64)
        env.close()

        # End of episode logging
        if cooperative:
            shared_reward = sum(total_rewards.values())
            total_cooperative_rewards.append(shared_reward)
            print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")
            #print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {shared_reward}")
        else:
            for agent in total_rewards:
                rewards_history[agent].append(total_rewards[agent])
            print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")

        # Convert records to a DataFrame and save to CSV
        df_eval = pd.DataFrame(data_records)

        # Ensure a directory for saving exists
        if not os.path.exists('evaluation_data'):
            os.makedirs('evaluation_data')

        # Save to CSV
        df_eval.to_csv('evaluation_data/training_data.csv', index=False)
    print(f"Training data saved to evaluation_data/training_data.csv")

    # Calculate average rewards for each agent or cooperative rewards
    if cooperative:
        avg_rewards = sum(total_cooperative_rewards) / len(total_cooperative_rewards)
    else:
        avg_rewards = {agent: sum(rewards) / len(rewards) for agent, rewards in rewards_history.items()}

    return avg_rewards

#09.09.24
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, action_size, cooperative=False, learning_rate=1e-3, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, min_epsilon=0.01):
        self.input_dim = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.memory = deque(maxlen=2000)
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.cooperative = cooperative

    # Manually added method to reshape tensors to avoid DimensionMismatch (old, misfunctional version)
    #def reshape_tensor(self, tensor, desired_shape):
    #    if tensor.shape != desired_shape:
    #        if tensor.shape[1] < desired_shape[1]:
    #            padding_size = desired_shape[1] - tensor.shape[1]
    #            padding = torch.zeros(tensor.shape[0], padding_size, dtype=tensor.dtype)
    #            tensor = torch.cat([tensor, padding], dim=1)
    #        elif tensor.shape[1] > desired_shape[1]:
    #            tensor = tensor[:, :desired_shape[1]]
    #    return tensor

    #def reshape_tensor(self, tensor, desired_shape):
        # Ensure that the number of dimensions is the same
    #    if tensor.dim() != len(desired_shape):
    #        raise ValueError(
    #            f"Tensor has {tensor.dim()} dimensions but desired shape requires {len(desired_shape)} dimensions.")

        # Process each dimension independently
    #    for i in range(len(desired_shape)):
    #        if tensor.shape[i] < desired_shape[i]:
    #            # Padding for the current dimension
    #            padding_size = desired_shape[i] - tensor.shape[i]
    #            pad_shape = list(tensor.shape)
    #            pad_shape[i] = padding_size
    #            padding = torch.zeros(*pad_shape, dtype=tensor.dtype)
    #            tensor = torch.cat([tensor, padding], dim=i)
    #        elif tensor.shape[i] > desired_shape[i]:
    #            # Trimming for the current dimension
    #            slices = [slice(None)] * len(tensor.shape)
    #            slices[i] = slice(0, desired_shape[i])
    #            tensor = tensor[tuple(slices)]

    #    return tensor

    # New, reformulated reshape_tensor() method
    def reshape_tensor(self, tensor, desired_shape):
        if len(desired_shape) == 2 and tensor.dim() > 2:
            # Flatten the tensor except for the batch dimension
            tensor = tensor.view(tensor.size(0), -1)

        if tensor.shape != desired_shape:
            if tensor.shape[1] < desired_shape[1]:
                padding_size = desired_shape[1] - tensor.shape[1]
                padding = torch.zeros(tensor.shape[0], padding_size, dtype=tensor.dtype)
                tensor = torch.cat([tensor, padding], dim=1)
            elif tensor.shape[1] > desired_shape[1]:
                tensor = tensor[:, :desired_shape[1]]
        return tensor

    # Old, initial act() method:
    #def act(self, state, other_agents=None):
    #    if np.random.rand() <= self.epsilon:
    #        return random.randrange(self.action_size)
    #    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    #    state = self.reshape_tensor(state, (1, self.input_dim))
    #    q_values = self.model(state)

    #    if self.cooperative and other_agents:
    #        combined_q_values = q_values.clone()
    #        for agent in other_agents:
    #            combined_q_values += agent.model(state)
    #        combined_q_values /= (1 + len(other_agents))
    #        return torch.argmax(combined_q_values).item()
    #    else:
    #        return torch.argmax(q_values).item()

    #Adjusted act() method:
    def act(self, state, other_agents=None):
        # Ensure action is within the valid range
        action = random.randrange(self.action_size)
        #Epsilon-greedy
        if np.random.rand() > self.epsilon:
            state = torch.tensor(state, dtype=torch.float32).clone().detach().unsqueeze(0)
            #state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            state = self.reshape_tensor(state, (1, self.input_dim))
            q_values = self.model(state)
            # if-check for cooperative behaviour, needs fine-tuning
            if self.cooperative and other_agents:
                combined_q_values = q_values.clone()
                for agent in other_agents:
                    combined_q_values += agent.model(state)
                combined_q_values /= (1 + len(other_agents))
                action = torch.argmax(combined_q_values).item()
            else:
                action = torch.argmax(q_values).item()
        # Add debug prints to ensure action is valid
        #print(f"Selected action: {action}")
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            state = self.reshape_tensor(state, (1, self.input_dim))
            #print(f"Shape of input tensor 'state': {state.shape}")

            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            next_state = self.reshape_tensor(next_state, (1, self.input_dim))
            #print(f"Shape of input tensor 'next_state': {next_state.shape}")

            assert state.shape[1] == self.input_dim, f"State dimension mismatch: {state.shape[1]} vs {self.input_dim}"
            assert next_state.shape[1] == self.input_dim, f"Next state dimension mismatch: {next_state.shape[1]} vs {self.input_dim}"

            #reward = reward.clone().detach().float().unsqueeze(0)
            reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
            #done = done.clone().detach().float().unsqueeze(0)
            done = torch.tensor(done, dtype=torch.float32).unsqueeze(0)
            target = reward

            if not done:
                with torch.no_grad():
                    next_state_value = self.model(next_state)
                    #print(f"Next state value shape: {next_state_value.shape}")
                target += self.gamma * torch.max(next_state_value)

            output = self.model(state)[0, action]

            # Ensure target is a tensor with the same shape as output
            target = target.clone().detach().float().unsqueeze(0)
            # target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
            #print(f"Shape of target tensor: {target.shape}")

            # Debug prints to ensure no NaNs or invalid values
            if torch.isnan(output).any() or torch.isnan(target).any():
                #print("NaN detected in output or target!")
                continue

            # Loss calculation
            loss = self.criterion(output.reshape(1,-1), target)

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Epsilon decay
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay


#11.09.24

import torch
import pandas as pd
import os
import time
from environments.pettingzoo_env2 import make_env


def evaluate(agents, num_episodes=10, cooperative=False):
    env = make_env()
    rewards_history = {agent: [] for agent in agents.keys()}  # Individual rewards
    total_cooperative_rewards = []  # Cooperative rewards
    data_records = []  # List for step-by-step evaluation data

    for episode in range(num_episodes):
        env.reset(seed=42)
        total_rewards = {agent: 0 for agent in env.possible_agents}
        episode_done = False

        while not episode_done:
            for agent in env.possible_agents:
                observation, reward, termination, truncation, _ = env.last()

                if termination or truncation:
                    env.step(None)  # No action for terminated agents
                    continue  # Don't break, just continue to the next agent

                obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    if cooperative:
                        action = agents[agent].act(obs_tensor)
                    else:
                        action = torch.argmax(agents[agent].model(obs_tensor)).item()

                env.step(action)

                # Update rewards for the current agent
                total_rewards[agent] += reward

                # Store step data
                next_observation, reward, termination, truncation, _ = env.last()
                data_records.append({
                    'Episode': episode + 1,
                    'Agent': agent,
                    'Mode': 'cooperative' if cooperative else 'individual',
                    'Action': action,
                    'Observation': observation,
                    'Next Observation': next_observation,
                    'Reward': reward,
                    'Total Reward': total_rewards[agent],
                    'Cooperative Reward': sum(total_rewards.values()) if cooperative else 0,
                    'Done': termination or truncation,
                })

            # Check if all agents are done
                episode_done = all([termination or truncation for agent in env.possible_agents])

        # Cooperative or individual rewards logging
        if cooperative:
            total_shared_reward = sum(total_rewards.values())
            total_cooperative_rewards.append(total_shared_reward)
        else:
            for agent in total_rewards:
                rewards_history[agent].append(total_rewards[agent])

        print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")

        # Save step-by-step evaluation data to CSV
        df_eval = pd.DataFrame(data_records)
        if not os.path.exists('evaluation_data'):
            os.makedirs('evaluation_data')
        df_eval.to_csv('evaluation_data/evaluation_data.csv', index=False)

    print(f"Evaluation data saved to evaluation_data/evaluation_data.csv")

    env.close()

    # Calculate average rewards
    if not cooperative:
        avg_rewards_individual = {agent: sum(rewards) / len(rewards) for agent, rewards in rewards_history.items()}
        avg_rewards_cooperative = None
    else:
        avg_rewards_individual = None
        avg_rewards_cooperative = sum(total_cooperative_rewards) / len(total_cooperative_rewards)

    return rewards_history, avg_rewards_individual, avg_rewards_cooperative

# Note: Die Antwort für die vom Zahlenwert 10 abweichenden Belohnungen liegt in der simple_tag.py:
# "Adversary reward can optionally be shaped (decreased reward for increased distance from agents)
# (Good) agent reward can " (increased reward for increased distance from adversary)."
# Note: Dass der "good_agent" nach wie vor positive Belohnungen (besonders +10) erhalten kann ist nach wie vor merkwürdig


    def update(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            state = self.reshape_tensor(state, (1, self.input_dim))
            #print(f"Shape of input tensor 'state': {state.shape}")

            #next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            next_state = next_state.clone().detach()

            next_state = self.reshape_tensor(next_state, (1, self.input_dim))
            #print(f"Shape of input tensor 'next_state': {next_state.shape}")

            assert state.shape[1] == self.input_dim, f"State dimension mismatch: {state.shape[1]} vs {self.input_dim}"
            assert next_state.shape[1] == self.input_dim, f"Next state dimension mismatch: {next_state.shape[1]} vs {self.input_dim}"

            #reward = reward.clone().detach().float().unsqueeze(0)
            reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
            #done = done.clone().detach().float().unsqueeze(0)
            done = torch.tensor(done, dtype=torch.float32).unsqueeze(0)
            target = reward.clone()

            if not done.item():
                with torch.no_grad():
                    next_state_value = self.model(next_state).max(1)[0]
                    #next_state_value = self.model(next_state)
                    #print(f"Next state value shape: {next_state_value.shape}")
                target = reward + (1 - done) * self.gamma * next_state_value
                #target += self.gamma * torch.max(next_state_value)
            output = self.model(state)[0, action]
            target = target.unsqueeze(1)
            output = output.unsqueeze(0)

            # Fix target shape if it is a scalar or has incompatible shape
            #if target.dim() == 1:  # If target is a vector
            #    target = target.expand(output.shape)
            #elif target.dim() == 0:  # If target is a scalar
            #    target = torch.full_like(output, target.item())

            # target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
            #print(f"Shape of target tensor: {target.shape}")

            # Ensure target and output are of the same shape
            #assert output.shape == target.shape, f"Output shape {output.shape} does not match target shape {target.shape}"

            # Debug prints to ensure no NaNs or invalid values
            if torch.isnan(output).any() or torch.isnan(target).any():
                print("NaN detected in output or target!")
                continue

            # Loss calculation
            loss = self.criterion(output, target)

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Epsilon decay
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay


