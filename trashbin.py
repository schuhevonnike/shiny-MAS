#Code läuft bis zur Initialisierung des Environments, dann aber Problem mit der Observation
from pettingzoo.mpe import simple_tag_v3
from pettingzoo.utils import wrappers

def make_env():
    # Load the simple_tag_v3 environment as ParallelEnv
    env = simple_tag_v3.env()

    # Apply wrappers that are compatible with AEC environments
    env = wrappers.OrderEnforcingWrapper(env)
    return env

if __name__ == "__main__":
    env = make_env()

    # Initialize the environment and get initial observations
    env.reset()
    print(f"Environment initialized")  # Debugging print to verify output

    #agent_iter() is a method used to iterate over all possible agents
    for agent in env.agent_iter():
        #Retrieve obs, rew and further info from the previous agent/step
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample()  # Take random actions

        # Execute the action
        env.step(action)  # Do not unpack, as step does not return anything

        if termination or truncation:
            env.reset()  # Reset the environment for the next episode

    env.render()  # Adjust render mode as needed
    env.close()

#Old evaluation2.py
import torch
from environments.pettingzoo_env2 import make_env


def evaluate(agents, num_episodes=100, cooperative=False):
    env = make_env()
    rewards_history = {agent: [] for agent in agents.keys()}

    for episode in range(num_episodes):
        observation = env.reset()
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

#Old training2.py:
def train(agents, num_episodes=100, cooperative=False):
    env = make_env()
    # Check if `env` is an environment instance and not a wrapper object
    if not hasattr(env, 'reset') or not hasattr(env, 'step'):
        raise TypeError("Provided env is not a valid environment instance")

    rewards_history = {agent: [] for agent in agents}
    gamma = 0.99  # Discount factor, adjustable based on algorithm
    batch_size = 64  # Batch size, relevant for algorithms with replay memory

    for episode in range(num_episodes):
        observation = env.reset()
        total_rewards = {agent: 0 for agent in env.possible_agents}
        done = False

        while not done:
            actions = {}
            for agent in env.agent_iter():
                obs = observation[agent]

                # Select action based on the mode (cooperative or individual)
                action = select_action(agents[agent], obs, cooperative, other_agents=agents.values())
                actions[agent] = action

            next_observation, rewards, done, _ = env.step(actions)

            # Collect rewards and update models
            for agent in env.possible_agents:
                reward = rewards[agent]
                next_obs = next_observation[agent]
                obs = observation[agent]

                # Store experience and update agent model
                agents[agent].remember(obs, action, reward, next_obs, done)

                if isinstance(agents[agent], DQNAgent):
                    # Update DQN agent
                    if len(agents[agent].memory) >= batch_size:
                        agents[agent].replay(batch_size)
                elif isinstance(agents[agent], MADDPGAgent):
                    # Update MADDPG agent
                    if len(agents[agent].memory) >= batch_size:
                        agents[agent].update(batch_size)
                elif isinstance(agents[agent], PPOAgent):
                    # Update PPO agent
                    if len(agents[agent].memory) >= batch_size:
                        agents[agent].update()
                elif isinstance(agents[agent], SACAgent):
                    # Update SAC agent
                    if len(agents[agent].memory) >= batch_size:
                        agents[agent].update(batch_size)

                total_rewards[agent] += reward

            observation = next_observation

            if done:
                break

        for agent in total_rewards:
            rewards_history[agent].append(total_rewards[agent])

        print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")

    env.close()
    return rewards_history



