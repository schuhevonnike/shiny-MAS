import numpy as np 

def evaluate(env, agents, episodes=100, cooperative=False):
    rewards = []
    for episode in range(episodes):
        state = env.reset()
        done = {agent: False for agent in env.agents}
        total_rewards = {agent: 0 for agent in env.agents}
        while not all(done.values()):
            actions = {}
            for agent in env.agents:
                if not done[agent]:
                    actions[agent] = agents[agent].select_action(state[agent], evaluation=True)
            state, rewards_step, done, _ = env.step(actions)
            for agent in env.agents:
                if not done[agent]:
                    total_rewards[agent] += rewards_step[agent]
        if cooperative:
            total_reward = sum(total_rewards.values())
            rewards.append(total_reward)
        else:
            for agent, reward in total_rewards.items():
                rewards.append(reward)
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")
    return mean_reward, std_reward
