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
                # "Replace torch.tensor(..., dtype=...) with target = target.clone().detach().float().unsqueeze(0)"
                # Convert float to PyTorch tensor
                obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                # Use tensor operations
                obs_tensor = obs_tensor.clone().detach().float().unsqueeze(0)
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

#Old training2.py:
import torch
from environments.pettingzoo_env2 import make_env
from algorithms01.dqn import DQNAgent
from algorithms01.maddpg import MADDPGAgent
from algorithms01.ppo import PPOAgent
from algorithms01.sac import SACAgent


def select_action(agent, observation, cooperative=False, other_agents=None):
    with torch.no_grad():
        if isinstance(agent, DQNAgent):
            action = agent.act(observation)
        elif isinstance(agent, PPOAgent):
            action, _ = agent.act(observation)
        elif isinstance(agent, SACAgent):
            action, _ = agent.act(observation)
        elif isinstance(agent, MADDPGAgent):
            action = agent.act(observation)
        else:
            raise ValueError("Unknown agent type")
    return action


def train(agents, num_episodes=10, cooperative=False):
    env = make_env()
    # Check if `env` is an environment instance and not a wrapper object
    if not hasattr(env, 'reset') or not hasattr(env, 'step'):
        raise TypeError("Provided env is not a valid environment instance")

    rewards_history = {agent: [] for agent in agents}
    gamma = 0.99  # Discount factor, adjustable based on algorithm
    batch_size = 64  # Batch size, relevant for algorithms with replay memory

    for episode in range(num_episodes):
        env.reset()
        total_rewards = {agent: 0 for agent in env.possible_agents}

        for agent in env.agent_iter():
            observation, reward, termination, truncation, _ = env.last()

            if termination or truncation:
                env.step(None)  # Tell the environment no action will be taken
                continue

            # Select action based on the mode (cooperative or individual)
            action = select_action(agents[agent], observation, cooperative, other_agents=agents.values())
            env.step(action)

            next_observation, reward, termination, truncation, _ = env.last()
            total_rewards[agent] += reward

            # Store experience and update agent model
            agents[agent].remember(observation, action, reward, next_observation, termination or truncation)

            if isinstance(agents[agent], DQNAgent):
                if len(agents[agent].memory) >= batch_size:
                    agents[agent].replay(batch_size)
            elif isinstance(agents[agent], MADDPGAgent):
                if len(agents[agent].memory) >= batch_size:
                    agents[agent].update(batch_size)
            elif isinstance(agents[agent], PPOAgent):
                if len(agents[agent].memory) >= batch_size:
                    agents[agent].update()
            elif isinstance(agents[agent], SACAgent):
                if len(agents[agent].memory) >= batch_size:
                    agents[agent].update(batch_size)

        # Log rewards
        for agent in total_rewards:
            rewards_history[agent].append(total_rewards[agent])

        print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")

    env.close()
    return rewards_history

#Another one

def train(agents, num_episodes=10, cooperative=False):
    env = make_env()
    # Check if `env` is an environment instance and not a wrapper object
    if not hasattr(env, 'reset') or not hasattr(env, 'step'):
        raise TypeError("Provided env is not a valid environment instance")

    rewards_history = {agent: [] for agent in agents}
    gamma = 0.99  # Discount factor, adjustable based on algorithm
    batch_size = 64  # Batch size, relevant for algorithms with replay memory

    for episode in range(num_episodes):
        env.reset()
        total_rewards = {agent: 0 for agent in env.possible_agents}
        print(f"Episode {episode + 1}/{num_episodes} started.")

        done = False  # Flag to manage episode completion

        while not done:
            for agent in env.agent_iter():
                observation, reward, termination, truncation, _ = env.last()
                print(f"Agent: {agent}, Observation: {observation}, Reward: {reward}")

                if termination or truncation:
                    done = True
                    env.step(None)  # Tell the environment no action will be taken
                    break  # Exit agent iteration if episode ended

                # Select action based on the mode (cooperative or individual)
                action = select_action(agents[agent], observation, cooperative, other_agents=agents.values())
                print(f"Selected Action for {agent}: {action}")

                env.step(action)

                next_observation, reward, termination, truncation, _ = env.last()
                total_rewards[agent] += reward

                # Store experience and update agent model
                agents[agent].remember(observation, action, reward, next_observation, termination or truncation)

                # Perform agent-specific updates
                if isinstance(agents[agent], DQNAgent):
                    if len(agents[agent].memory) >= batch_size:
                        agents[agent].replay(batch_size)
                elif isinstance(agents[agent], MADDPGAgent):
                    if len(agents[agent].memory) >= batch_size:
                        agents[agent].update(batch_size)
                elif isinstance(agents[agent], PPOAgent):
                    if len(agents[agent].memory) >= batch_size:
                        agents[agent].update()
                elif isinstance(agents[agent], SACAgent):
                    if len(agents[agent].memory) >= batch_size:
                        agents[agent].update(batch_size)

        # Log rewards
        for agent in total_rewards:
            rewards_history[agent].append(total_rewards[agent])
        print(f"Episode {episode + 1} completed with Total Rewards: {total_rewards}")

    env.close()
    return rewards_history