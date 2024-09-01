import torch
from environments.pettingzoo_env2 import parallel_env


def select_action(policy, observation, cooperative=False):
    with torch.no_grad():
        observation = torch.tensor(observation, dtype=torch.float32)
        action_probs = policy(observation)
        if cooperative:
            action = torch.argmax(action_probs).item()
        else:
            action = action_probs.multinomial(num_samples=1).item()
    return action


def train(agents, num_episodes=1000, gamma=0.99, cooperative=False):
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

            action = select_action(agents[agent], observation, cooperative)
            env.step(action)

            next_observation, reward, done, _ = env.last(observation)
            reward = sum(reward.values()) if cooperative else reward[agent]

            # Update the model
            target = reward + gamma * torch.max(agents[agent](torch.tensor(next_observation, dtype=torch.float32)))
            predicted = agents[agent](torch.tensor(observation, dtype=torch.float32))[action]
            loss = torch.nn.functional.mse_loss(predicted, target)

            agents[agent].optimizer.zero_grad()
            loss.backward()
            agents[agent].optimizer.step()

            total_rewards[agent] += reward

            if done:
                break

        for agent in total_rewards:
            rewards_history[agent].append(total_rewards[agent])

        print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")

    env.close()
    return rewards_history
