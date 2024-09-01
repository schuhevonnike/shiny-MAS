import torch
from environments.pettingzoo_env2 import parallel_env


def evaluate(agents, num_episodes=100, cooperative=False):
    env = parallel_env()
    rewards_history = {agent: [] for agent in agents.keys()}

    for episode in range(num_episodes):
        observations = env.reset()
        total_rewards = {agent: 0 for agent in env.possible_agents}

        for agent in env.agent_iter():
            observation, reward, termination, truncation, _ = env.last()

            if termination or truncation:
                env.step(None)
                continue

            with torch.no_grad():
                observation = torch.tensor(observation, dtype=torch.float32)
                action = torch.argmax(agents[agent](observation)).item()
            env.step(action)

            reward = sum(reward.values()) if cooperative else reward[agent]
            total_rewards[agent] += reward

            if all(termination.values()):
                break

        for agent in total_rewards:
            rewards_history[agent].append(total_rewards[agent])

        print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")

    env.close()
    avg_rewards = {agent: sum(rewards) / num_episodes for agent, rewards in rewards_history.items()}
    return avg_rewards
