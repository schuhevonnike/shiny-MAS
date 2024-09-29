import os
import random
import numpy as np
import pandas as pd
import torch
from utils.pettingzoo_env import make_env

def train(agents, num_episodes):
    env = make_env()
    rewards_history = {agent: [] for agent in agents}
    data_records = []

    for episode in range(num_episodes):
        env.reset()
        #env.reset(seed=42)
        total_rewards = {agent: 0 for agent in env.possible_agents}
        # Initialize done flag for tracking done state for each agent.
        done = {agent: False for agent in env.possible_agents}

        # Initialize storage for each agent's last observation and action
        last_observation = {agent: None for agent in env.possible_agents}
        last_action = {agent: None for agent in env.possible_agents}

        for agent in env.agent_iter():
            observation, reward, termination, truncation, _ = env.last()
            current_done = termination or truncation
            done[agent] = current_done

            # Update total rewards
            total_rewards[agent] += reward

            # If it's not the first turn for the agent, store the transition
            if last_observation[agent] is not None and last_action[agent] is not None:
                agents[agent].remember(
                    #state=last_observation[agent].copy(),
                    state=last_observation[agent],
                    action=last_action[agent],
                    reward=reward,
                    next_state=observation.copy(),
                    done=current_done
                )

            # Select action
            if not current_done:
                # Version 1 of action selection - a loop with close resemblance to the DQN variant
                with torch.no_grad():
                    obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                    # Select an action based on the current policy.
                    #action = agents[agent].actor(obs_tensor).detach().numpy()[0]
                    action = agents[agent].actor(obs_tensor).squeeze(0).numpy()
                agents[agent].actor.train()
                action += 0.1 * np.random.randn(*action.shape)  # Add exploration noise
                #return np.clip(action, -1, 1)  # Assuming action space [-1, 1]

                # Version 2 of action selection
                # Select an action based on the current policy.
                #observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                #action = agents[agent].actor(observation).detach().numpy()[0]
                #action += 0.1 * np.random.randn(*action.shape)  # Add exploration noise (float = 0.1)
                #return np.clip(action, -1, 1)  # Assuming action space [-1, 1]
            else:
                action = None

            last_observation[agent] = observation
            last_action[agent] = action

            # Log the step data in a pd.df
            data_records.append({
                'Episode': episode,
                'Agent': agent,
                'Action': action,
                'Observation': observation,
                'Reward': reward,
                'Done': termination or truncation,
            })

            # Take a step in the environment
            env.step(action)

        # Update agents
        for agent in agents:
            other_agent = next(a for a in agents.values() if a != agents[agent])
            agents[agent].update(other_agent)

        # Logging rewards at the end of each episode
        for agent in total_rewards:
            rewards_history[agent].append(total_rewards[agent])
        print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")
        print(f"Mean Reward last 100 Episodes:")
        for agent in agents:
            #print(f"Agent {agent} Reward: {np.array(rewards_history[agent][-100:]).mean()}")
            print(f"Agent {agent} Reward: {np.mean(rewards_history[agent][-100:])}")
        print()

        # Save the recorded data to a CSV
        df_eval = pd.DataFrame(data_records)

        if not os.path.exists('data_exportMADDPG'):
            os.makedirs('data_exportMADDPG')

        df_eval.to_csv('data_exportMADDPG/training_data.csv', index=False)
        print(f"Training data saved to data_exportMADDPG/training_data.csv")

    avg_rewards = {agent: sum(rewards) / len(rewards) for agent, rewards in rewards_history.items()}
    #avg_rewards = {agent: np.mean(rewards) for agent, rewards in rewards_history.items()}
    print(f"Type of avg_rewards in train: {type(avg_rewards)}") # Debugging
    return avg_rewards

def evaluate(agents, num_episodes):
    env = make_env()
    rewards_history = {agent: [] for agent in agents}
    data_records = []

    for episode in range(num_episodes):
        env.reset()
        #env.reset(seed=42)
        total_rewards = {agent: 0 for agent in env.possible_agents}
        # Initialize done flag for tracking done state for each agent.
        done = {agent: False for agent in env.possible_agents}

        # Initialize storage for each agent's last observation and action
        last_observation = {agent: None for agent in env.possible_agents}
        last_action = {agent: None for agent in env.possible_agents}

        for agent in env.agent_iter():
            observation, reward, termination, truncation, _ = env.last()
            current_done = termination or truncation
            done[agent] = current_done

            # Update total rewards
            total_rewards[agent] += reward

            # If it's not the first turn for the agent, store the transition
            if last_observation[agent] is not None and last_action[agent] is not None:
                agents[agent].remember(
                    state=last_observation[agent].copy(),
                    #state=last_observation[agent],
                    action=last_action[agent],
                    reward=reward,
                    next_state=observation.copy(),
                    done=current_done
                )

            # Select action
            if not current_done:
                # Loop with close resemblance to the DQN variant
                with torch.no_grad():
                    obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                    # Select an action based on the current policy.
                    #action = agents[agent].actor(obs_tensor).detach().numpy()[0]
                    action = agents[agent].actor(obs_tensor).squeeze(0).numpy()
                agents[agent].actor.train()
                #return np.clip(action, -1, 1)  # Assuming action space [-1, 1]

            else:
                action = None

            last_observation[agent] = observation
            last_action[agent] = action

            # Log the step data in a pd.df
            data_records.append({
                'Episode': episode,
                'Agent': agent,
                'Action': action,
                'Observation': observation,
                'Reward': reward,
                'Done': termination or truncation,
            })

            # Take a step in the environment
            env.step(action)

        # Update agents
        for agent in agents:
            other_agent = next(a for a in agents.values() if a != agents[agent])
            agents[agent].update(other_agent)

        # Logging rewards at the end of each episode - modified 26.09.24
        for agent in total_rewards:
            rewards_history[agent].append(total_rewards[agent])
        print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")
        print(f"Mean Reward last 100 Episodes:")
        for agent in agents:
            print(f"Agent {agent} Reward: {np.array(rewards_history[agent][-100:]).mean()}")
            print(f"Agent {agent} Reward: {np.mean(rewards_history[agent][-100:])}")
        print()

        # Save the recorded data to a CSV
        df_eval = pd.DataFrame(data_records)

        if not os.path.exists('data_exportMADDPG'):
            os.makedirs('data_exportMADDPG')

        df_eval.to_csv('data_exportMADDPG/evaluation_data.csv', index=False)
        print(f"Evaluation data saved to data_exportMADDPG/training_data.csv")

    avg_rewards = {agent: sum(rewards) / len(rewards) for agent, rewards in rewards_history.items()}
    # avg_rewards = {agent: np.mean(rewards) for agent, rewards in rewards_history.items()}
    print(f"Type of avg_rewards in train: {type(avg_rewards)}")  # Debugging
    return avg_rewards