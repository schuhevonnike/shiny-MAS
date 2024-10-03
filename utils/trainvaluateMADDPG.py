import os
import numpy as np
import pandas as pd
import torch
from utils.pettingzoo_env import make_env

def train(agents, num_episodes, seed):
    env = make_env()
    rewards_history = {agent: [] for agent in agents}
    data_records = []

    for episode in range(num_episodes):
        env.reset(seed=seed)
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
                with torch.no_grad():
                    # Gather current tau value
                    tau = agents[agent].tau
                    obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                    # print(f"Observation tensor: {obs_tensor}")          # Debugging
                    # print(f"Last action tensor: {last_action_tensor}")  # Debugging
                    action_probs = agents[agent].actor(obs_tensor).detach()
                    # Apply softmax to convert logits to probabilities, add exploration noise - divide action_probs by tau before converting
                    action_probs = torch.softmax(action_probs/tau, -1)
                    # print(f"Action probs: {action_probs}")

                    # Note: all three ways of collecting/sampling actions works with the one-hot encoding
                    action = torch.argmax(action_probs).item()
                    # action = torch.multinomial(action_probs.view(-1),1).item()
                    # action = torch.multinomial(action_probs.squeeze(0), 1).item()
                    # print(f"Action: {action}")

                    # Convert action to one-hot encoding for shape [32, 5]
                    one_hot_action = torch.zeros(5)  # Assuming 5 possible actions
                    one_hot_action[action] = 1.0
                    # print(f"One-hot action: {one_hot_action}")
            else:
                action = None

            last_observation[agent] = observation
            last_action[agent] = one_hot_action

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

        for agent_id in agents:
            # Update exploration parameter tau
            agents[agent_id].tau = max(agents[agent_id].tau * agents[agent_id].tau_decay, agents[agent_id].tau_min)
            # Update agents memory, if necessary
            if len(agents[agent_id].memory) >= 256:
                for other_agent_id in agents:
                    if other_agent_id != agent_id:  # Ensure we are updating with another agent
                        agents[agent_id].update(other_agent_id, agents)

        # Logging rewards at the end of each episode
        for agent in total_rewards:
            rewards_history[agent].append(total_rewards[agent])
        print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")
        print(f"Mean Reward last 100 Episodes:")
        for agent in agents:
            print(f"Agent {agent} Reward: {np.array(rewards_history[agent][-100:]).mean()}")
            # print(f"Agent {agent} Reward: {np.mean(rewards_history[agent][-100:])}")
        print()

        # Save the recorded data to a CSV
        df_eval = pd.DataFrame(data_records)

        if not os.path.exists('data_exportMADDPG'):
            os.makedirs('data_exportMADDPG')

        df_eval.to_csv(f'data_exportMADDPG/training_data_{seed}.csv', index=False)
        print(f"\nTraining data saved to data_exportMADDPG/training_data.csv")

    avg_rewards = {agent: sum(rewards) / len(rewards) for agent, rewards in rewards_history.items()}
    #avg_rewards = {agent: np.mean(rewards) for agent, rewards in rewards_history.items()}
    # print(f"Type of avg_rewards in train: {type(avg_rewards)}") # Debugging
    return avg_rewards

def evaluate(agents, num_episodes, seed):
    env = make_env()
    rewards_history = {agent: [] for agent in agents}
    data_records = []

    for episode in range(num_episodes):
        env.reset(seed=seed)
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
                with torch.no_grad():
                    obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                    action_probs = agents[agent].actor(obs_tensor).detach()
                    # Apply softmax to convert logits to probabilities
                    action_probs = torch.softmax(action_probs, -1)

                    # Note: all three ways of collecting/sampling actions works with the one-hot encoding
                    action = torch.argmax(action_probs).item()
                    # action = torch.multinomial(action_probs.view(-1),1).item()
                    # action = torch.multinomial(action_probs.squeeze(0), 1).item()

                    # Exploration off for evaluating

                    # Convert action to one-hot encoding for shape [32, 5]
                    one_hot_action = torch.zeros(5)  # Assuming 5 possible actions
                    one_hot_action[action] = 1.0
            else:
                action = None

            last_observation[agent] = observation
            last_action[agent] = one_hot_action

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
        for agent_id in agents:
            if len(agents[agent_id].memory) >= 256:
                for other_agent_id in agents:
                    if other_agent_id != agent_id:  # Ensure we are updating with another agent
                        agents[agent_id].update(other_agent_id, agents)

        # Logging rewards at the end of each episode - modified 26.09.24
        for agent in total_rewards:
            rewards_history[agent].append(total_rewards[agent])
        print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")
        print(f"Mean Reward last 100 Episodes:")
        for agent in agents:
            print(f"Agent {agent} Reward: {np.array(rewards_history[agent][-100:]).mean()}")
            # print(f"Agent {agent} Reward: {np.mean(rewards_history[agent][-100:])}")
        print()

        # Save the recorded data to a CSV
        df_eval = pd.DataFrame(data_records)

        if not os.path.exists('data_exportMADDPG'):
            os.makedirs('data_exportMADDPG')

        df_eval.to_csv(f'data_exportMADDPG/evaluation_data_{seed}.csv', index=False)
        print(f"\nEvaluation data saved to data_exportMADDPG/training_data.csv")

    avg_rewards = {agent: sum(rewards) / len(rewards) for agent, rewards in rewards_history.items()}
    #avg_rewards = {agent: np.mean(rewards) for agent, rewards in rewards_history.items()}
    # print(f"Type of avg_rewards in train: {type(avg_rewards)}")  # Debugging
    return avg_rewards