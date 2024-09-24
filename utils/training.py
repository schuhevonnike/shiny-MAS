#import os
#import pandas as pd
import random

import numpy as np
import torch
from environments.pettingzoo_env import make_env

def select_action(agent, observation):
    with torch.no_grad:
        action = agent.act(observation)
    return action


def train(agents, num_episodes):
    env = make_env()
    rewards_history = {agent: [] for agent in agents}

    for episode in range(num_episodes):
        env.reset()
        total_rewards = {agent: 0 for agent in env.possible_agents}
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
                if random.random() > agents[agent].epsilon:
                    with torch.no_grad():
                        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                        q_values = agents[agent].model(obs_tensor)
                        action = torch.argmax(q_values).item()
                else:
                    action = env.action_space(agent).sample()
            else:
                action = None

            last_observation[agent] = observation
            last_action[agent] = action
            # Step the environment
            env.step(action)
        for agent in agents:
            if len(agents[agent].memory) >= 256:
                for i in range(100):
                    agents[agent].update(256)

        for agent in total_rewards:
            rewards_history[agent].append(total_rewards[agent])
        print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")
        print(f"Mean Reward last 100 Episodes:")
        for agent in agents:
            print(f"Agent {agent} Reward: {np.array(rewards_history[agent][-100:]).mean()}")
        print()

    avg_rewards = {agent: sum(rewards) / len(rewards) for agent, rewards in rewards_history.items()}
    return avg_rewards