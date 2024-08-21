import numpy as np
import torch
from torch.optim import Adam

def train(env, adversary_agents, cooperator_agents, num_episodes):
    for episode in range(num_episodes):
        # Initialize the state
        state = env.reset()
        done = {agent: False for agent in env.agents}
        adversary_rewards = {agent: 0 for agent in adversary_agents}
        cooperator_rewards = {agent: 0 for agent in cooperator_agents}

        while not all(done.values()):
            actions = {}
            # Determine actions for all agents
            for agent in env.agents:
                if agent in adversary_agents:
                    actions[agent] = adversary_agents[agent].select_action(state[0][agent])
                elif agent in cooperator_agents:
                    actions[agent] = cooperator_agents[agent].select_action(state[0][agent])

            # Perform a step in the environment
            obs, rewards, done, infos = env.step(actions)
            # Update the state with the latest observations for the next iteration
            state = [obs]  # Make sure to structure `state` correctly for the next iteration

            # Update agents based on the results of the step
            for agent in env.agents:
                if agent in adversary_agents and not done[agent]:
                    adversary_agents[agent].update(state[0][0][agent], actions[agent], rewards[agent], obs[agent], done[agent])
                    adversary_rewards[agent] += rewards[agent]
                elif agent in cooperator_agents and not done[agent]:
                    cooperator_agents[agent].update(state[0][0][agent], actions[agent], rewards[agent], obs[agent], done[agent])
                    cooperator_rewards[agent] += rewards[agent]
            # After the loop completes, return the final observations, rewards, done flags, and infos
            return list(obs), list(rewards), list(done), infos

        adversary_total_reward = sum(adversary_rewards.values())
        cooperator_total_reward = sum(cooperator_rewards.values())
        print(
            f"Episode {episode + 1} - Adversary Total Reward: {adversary_total_reward}, Cooperator Total Reward: {cooperator_total_reward}")
        return adversary_agents, cooperator_agents
