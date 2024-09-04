import time
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
    rewards_history = {agent: [] for agent in agents}
    gamma = 0.99  # Discount factor, adjustable based on algorithm
    batch_size = 64  # Batch size, relevant for algorithms with replay memory
    # Check if `env` is an environment instance and not a wrapper object
    if not hasattr(env, 'reset') or not hasattr(env, 'step'):
        raise TypeError("Provided env is not a valid environment instance")

    for episode in range(num_episodes):
        episode_start_time = time.time()
        env.reset()
        total_rewards = {agent: 0 for agent in env.possible_agents}
        print(f"Episode {episode + 1}/{num_episodes} started.")

        done = False  # Flag to manage episode completion
        try:
            while not done:
                for agent in env.agent_iter():
                    step_start_time = time.time()
                    # Problem: Agent_0's Beobachtungen haben die Dimension 1x14 anstatt 1x16 wie die übrigen Agenten (Adversaries)
                    observation, reward, termination, truncation, _ = env.last()
                    print(f"Agent: {agent}, Observation: {observation}, Reward: {reward}")

                    if termination or truncation:
                        done = True
                        env.step(None)  # Tell the environment no action will be taken
                        break  # Exit agent iteration if episode ended

                    # Select action based on the mode (cooperative or individual)
                    action = select_action(agents[agent], observation, cooperative, other_agents=agents.values())
                    # action enthält tatsächlich eine Handlung des Agenten basierend auf dem Modus und seinen Beobachtungen
                    print(f"Selected Action for {agent}: {action}")

                    env.step(action)

                    next_observation, reward, termination, truncation, _ = env.last()
                    total_rewards[agent] += reward

                    # Timing the agent's memory and update operations
                    agent_start_time = time.time()

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
                        # Time taken for agent operations
                        print(f"Agent {agent} operation took {time.time() - agent_start_time:.2f} seconds")
                    # Time taken for a single environment step
                    print(f"Step operation took {time.time() - step_start_time:.2f} seconds")
                # Timing the end of an episode
                print(f"Episode {episode + 1} completed in {time.time() - episode_start_time:.2f} seconds with Total Rewards: {total_rewards}")
                # Log rewards
            for agent in total_rewards:
                rewards_history[agent].append(total_rewards[agent])

        except KeyboardInterrupt:
            print("Training interrupted by user.")
        finally:
            env.close()
    return rewards_history