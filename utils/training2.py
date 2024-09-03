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