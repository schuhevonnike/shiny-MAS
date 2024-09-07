import torch
import pandas as pd
from environments.pettingzoo_env2 import make_env

def evaluate(agents, num_episodes=10, cooperative=False):
    env = make_env()
    rewards_history = {agent: [] for agent in agents.keys()}
    data_records = {agent: [] for agent in agents.keys()} # Dicts for later evaluation with pandas

    for episode in range(num_episodes):
        env.reset(seed=42)
        total_rewards = {agent: 0 for agent in env.possible_agents}
        done = False  # Flag to manage episode completion

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
                    # Aggregate total rewards from all agents for cooperative mode
                    shared_reward = sum(sum(rewards) for rewards in rewards_history.values())
                    total_rewards[agent] += shared_reward
                else:
                    total_rewards[agent] += reward

        # Note: Die Antwort für die vom Zahlenwert 10 abweichenden Belohnungen liegt in der simple_tag.py:
        # "Adversary reward can optionally be shaped (decreased reward for increased distance from agents)
        # (Good) agent reward can " (increased reward for increased distance from adversary)."
        # Note: Dass der "good_agent" nach wie vor positive Belohnungen (besonders +10) erhalten kann ist nach wie vor merkwürdig

        # Store total rewards per agent after each episode
        for agent in total_rewards:
            rewards_history[agent].append(total_rewards[agent])

        print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")

    env.close()

    # Calculate average rewards for each agent
    #Something about this is fishy
    avg_rewards = {agent: sum(rewards) / len(rewards) for agent, rewards in rewards_history.items()}
    return rewards_history, avg_rewards
