import numpy as np

def evaluate(env, adversary_agents, cooperator_agents, num_episodes):
    adversary_rewards = []
    cooperator_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = {agent: False for agent in env.agents}
        adversary_total_rewards = {agent: 0 for agent in adversary_agents}
        cooperator_total_rewards = {agent: 0 for agent in cooperator_agents}

        while not all(done.values()):
            actions = {}
            for agent in env.agents:
                if agent in adversary_agents:
                    actions[agent] = adversary_agents[agent].select_action(state[agent], evaluation=True)
                elif agent in cooperator_agents:
                    actions[agent] = cooperator_agents[agent].select_action(state[agent], evaluation=True)

            next_state, rewards, done, _ = env.step(actions)
            for agent in env.agents:
                if agent in adversary_agents and not done[agent]:
                    adversary_total_rewards[agent] += rewards[agent]
                elif agent in cooperator_agents and not done[agent]:
                    cooperator_total_rewards[agent] += rewards[agent]

        adversary_total_reward = sum(adversary_total_rewards.values())
        cooperator_total_reward = sum(cooperator_total_rewards.values())
        adversary_rewards.append(adversary_total_reward)
        cooperator_rewards.append(cooperator_total_reward)

    adversary_mean_reward = np.mean(adversary_rewards)
    adversary_std_reward = np.std(adversary_rewards)
    cooperator_mean_reward = np.mean(cooperator_rewards)
    cooperator_std_reward = np.std(cooperator_rewards)

    print(f"Adversary Mean Reward: {adversary_mean_reward}, Adversary Std Reward: {adversary_std_reward}")
    print(f"Cooperator Mean Reward: {cooperator_mean_reward}, Cooperator Std Reward: {cooperator_std_reward}")

    return adversary_mean_reward, adversary_std_reward, cooperator_mean_reward, cooperator_std_reward