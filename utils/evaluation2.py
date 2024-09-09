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

