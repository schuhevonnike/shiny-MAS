import numpy as np
import torch
from torch.optim import Adam

def train(env, agents, num_episodes, learning_rate=0.001, cooperative=False):
    #optimizers = [Adam(agent.parameters(), lr=learning_rate) for agent in agents]
    for episode in range(num_episodes):
        state = env.reset()
        done = {agent: False for agent in env.agents}
        total_rewards = {agent: 0 for agent in env.agents}
        while not all(done.values()):
            actions = {}
            for agent in env.agents:
                if not done[agent]:
                    actions[agent] = agents[agent].select_action(state[agent])
            next_state, rewards, done, _ = env.step(actions)
            for agent in env.agents:
                if not done[agent]:
                    agents[agent].update(state[agent], actions[agent], rewards[agent], next_state[agent], done[agent])
                    state[agent] = next_state[agent]
                    total_rewards[agent] += rewards[agent]
        if cooperative:
            total_reward = sum(total_rewards.values())
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")
        else:
            for agent, reward in total_rewards.items():
                print(f"Episode {episode + 1} - Agent {agent}: Total Reward: {reward}")
    return agents


