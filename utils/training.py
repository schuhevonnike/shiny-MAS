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

        # Assuming env.agents is something like ['agent_0', 'agent_1', 'agent_2', ...]
        agent_to_idx = {agent: idx for idx, agent in enumerate(env.agents)}
        while not all(done.values()):
            actions = {}
            for agent in env.agents:
                agent_idx = agent_to_idx[agent]  # Get the integer index for the current agent
                if agent in adversary_agents:
                    actions[agent] = adversary_agents[agent].select_action(state[agent_idx])
                elif agent in cooperator_agents:
                    actions[agent] = cooperator_agents[agent].select_action(state[agent_idx])
            obs, rewards, done, infos = env.step(actions)

            # Update agents based on the results of the step
            for agent in env.agents:
                agent_idx = agent_to_idx[agent]
                if agent in adversary_agents and not done[agent]:
                    adversary_agents[agent].update(state[agent_idx], actions[agent], rewards[agent], obs[agent], done[agent])
                    adversary_rewards[agent] += rewards[agent]
                elif agent in cooperator_agents and not done[agent]:
                    cooperator_agents[agent].update(state[agent_idx], actions[agent], rewards[agent], obs[agent], done[agent])
                    cooperator_rewards[agent] += rewards[agent]

            # Update the state for the next iteration
            # Make sure to structure `state` correctly as an array for further use
            state = [obs]

            #state = np.array([obs], dtype=np.float32)

            # After the loop completes, return the final observations, rewards, done flags, and infos
            return state, list(obs), list(rewards), list(done), infos

        adversary_total_reward = sum(adversary_rewards.values())
        cooperator_total_reward = sum(cooperator_rewards.values())
        print(
            f"Episode {episode + 1} - Adversary Total Reward: {adversary_total_reward}, Cooperator Total Reward: {cooperator_total_reward}")
        return adversary_agents, cooperator_agents
