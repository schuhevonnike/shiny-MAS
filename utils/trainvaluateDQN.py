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
                    state=last_observation[agent].copy(),
                    action=last_action[agent],
                    reward=reward,
                    next_state=observation.copy(),
                    done=current_done
                )

            # Select action
            if not current_done:
                # Epsilon greedy implementation
                if random.random() > agents[agent].epsilon:
                    with torch.no_grad():
                        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                        q_values = agents[agent].model(obs_tensor)
                        action = torch.argmax(q_values).item()
                else:
                    action = env.action_space(agent).sample()
            else:
                action = None
            print(action)
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

        for agent in agents:
            if len(agents[agent].memory) >= 256:
                for i in range(100):
                    agents[agent].update(256)

        # Logging rewards at the end of each episode - modified 26.09.24
        for agent in total_rewards:
            rewards_history[agent].append(total_rewards[agent])
        print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")
        print(f"Mean Reward last 100 Episodes:")
        for agent in agents:
            print(f"Agent {agent} Reward: {np.array(rewards_history[agent][-100:]).mean()}")
        print()

        # Save the recorded data to a CSV
        df_eval = pd.DataFrame(data_records)

        if not os.path.exists('data_exportDQN'):
            os.makedirs('data_exportDQN')

        df_eval.to_csv('data_exportDQN/training_data.csv', index=False)
        print(f"Training data saved to data_exportDQN/training_data.csv")

    avg_rewards = {agent: sum(rewards) / len(rewards) for agent, rewards in rewards_history.items()}
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
                    action=last_action[agent],
                    reward=reward,
                    next_state=observation.copy(),
                    done=current_done
                )

            # Select action
            if not current_done:
                with torch.no_grad():
                    obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                    q_values = agents[agent].model(obs_tensor)
                    action = torch.argmax(q_values).item()
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

        for agent in agents:
            if len(agents[agent].memory) >= 256:
                for i in range(100):
                    agents[agent].update(256)

        # Logging rewards at the end of each episode - modified 26.09.24
        for agent in total_rewards:
            rewards_history[agent].append(total_rewards[agent])
        print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")
        print(f"Mean Reward last 100 Episodes:")
        for agent in agents:
            print(f"Agent {agent} Reward: {np.array(rewards_history[agent][-100:]).mean()}")
        print()

        # Save the recorded data to a CSV
        df_eval = pd.DataFrame(data_records)

        if not os.path.exists('data_exportDQN'):
            os.makedirs('data_exportDQN')

        df_eval.to_csv('data_exportDQN/evaluation_data.csv', index=False)
        print(f"Evaluation data saved to data_exportDQN/training_data.csv")

    avg_rewards = {agent: sum(rewards) / len(rewards) for agent, rewards in rewards_history.items()}
    return avg_rewards