import numpy as np
import torch
from sympy import false
from torch.optim import Adam

def train(env, adversary_agents, cooperator_agents, num_episodes):
    for episode in range(num_episodes):
        state = env.reset()
        done = {agent: False for agent in env.agents}
        adversary_rewards = {agent: 0 for agent in adversary_agents}
        cooperator_rewards = {agent: 0 for agent in cooperator_agents}

        while not all(done.values()):
            actions = {}
            for agent in env.agents:
                if agent in adversary_agents:
                    actions[agent] = adversary_agents[agent].select_action(state[agent])
                elif agent in cooperator_agents:
                    actions[agent] = cooperator_agents[agent].select_action(state[agent])

            next_state, rewards, done, _ = env.step(actions)
            for agent in env.agents:
                if agent in adversary_agents and not done[agent]:
                    adversary_agents[agent].update(state[agent], actions[agent], rewards[agent], next_state[agent], done[agent])
                    adversary_rewards[agent] += rewards[agent]
                elif agent in cooperator_agents and not done[agent]:
                    cooperator_agents[agent].update(state[agent], actions[agent], rewards[agent], next_state[agent], done[agent])
                    cooperator_rewards[agent] += rewards[agent]

        adversary_total_reward = sum(adversary_rewards.values())
        cooperator_total_reward = sum(cooperator_rewards.values())
        print(f"Episode {episode + 1} - Adversary Total Reward: {adversary_total_reward}, Cooperator Total Reward: {cooperator_total_reward}")

    return adversary_agents, cooperator_agents