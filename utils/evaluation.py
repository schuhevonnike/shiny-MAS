#import os
#import pandas as pd
import torch
from environments.pettingzoo_env import make_env

def evaluate(agents, num_episodes):
    env = make_env()
    rewards_history = {agent: [] for agent in agents}
    #data_records = []

    # Currently not considered with this logic is the fact that the greedy-factor epsilon (in the case of DQN) needs to be set to 0.
    for episode in range(num_episodes):
        env.reset()
        total_rewards = {agent: 0 for agent in env.possible_agents}
        done = {agent: False for agent in env.possible_agents}
        while not all(done.values()):
            for agent in env.agent_iter():
                if done[agent]:
                    env.step(None)
                    continue
                observation, reward, termination, truncation, _ = env.last()
                if not (termination or truncation):
                    obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        action = torch.argmax(agents[agent].model(obs_tensor)).item()
                else:
                    action = None
                total_rewards[agent] += reward
                env.step(action)
                next_observation, reward, termination, truncation, _ = env.last()
                next_obs_tensor = torch.tensor(next_observation, dtype=torch.float32).unsqueeze(0)
                if termination or truncation:
                    done[agent] = True
                agents[agent].remember(observation, action, reward, next_obs_tensor, termination or truncation)
                if len(agents[agent].memory) >= 64:
                    agents[agent].update(64)
        env.close()
        for agent in total_rewards:
            rewards_history[agent].append(total_rewards[agent])
        print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")
    avg_rewards = {agent: sum(rewards) / len(rewards) for agent, rewards in rewards_history.items()}
    return avg_rewards