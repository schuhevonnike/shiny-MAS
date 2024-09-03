import torch
from environments.pettingzoo_env2 import make_env

def evaluate(agents, num_episodes=10, cooperative=False):
    env = make_env()
    rewards_history = {agent: [] for agent in agents.keys()}

    for episode in range(num_episodes):
        env.reset()
        total_rewards = {agent: 0 for agent in env.possible_agents}

        for agent in env.agent_iter():
            observation, reward, termination, truncation, _ = env.last()

            if termination or truncation:
                env.step(None)  # Step with None to signify no action taken for ended agent
                continue

            # Select action based on the mode (cooperative or individual)
            with torch.no_grad():
                obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                if cooperative:
                    # Cooperative mode: choose action considering cooperative strategy
                    action = agents[agent].act(obs_tensor)
                else:
                    # Individual mode: choose action independently
                    action = torch.argmax(agents[agent].model(obs_tensor)).item()

            env.step(action)

            # Update total rewards for this agent
            if cooperative:
                shared_reward = sum(reward.values())
                for a in total_rewards:
                    total_rewards[a] += shared_reward
            else:
                total_rewards[agent] += reward

        # Store total rewards per agent after each episode
        for agent in total_rewards:
            rewards_history[agent].append(total_rewards[agent])

        print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")

    env.close()

    # Calculate average rewards for each agent
    avg_rewards = {agent: sum(rewards) / len(rewards) for agent, rewards in rewards_history.items()}
    return avg_rewards
