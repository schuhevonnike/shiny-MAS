import os
import torch
import pandas as pd
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
    data_records = []  # List to store records per step
    total_cooperative_rewards = []  # Track cooperative rewards

    if not hasattr(env, 'reset') or not hasattr(env, 'step'):
        raise TypeError("Provided env is not a valid environment instance")

    for episode in range(num_episodes):
        env.reset(seed=42)
        total_rewards = {agent: 0 for agent in env.possible_agents}
        done = {agent: False for agent in env.possible_agents}  # Track done state for each agent
        step = 0  # Initialize step counter for each episode

        while not all(done.values()):  # Continue until all agents are done
            step += 1
            for agent in env.agent_iter():
                if done[agent]:  # If the agent is done, skip to next agent
                    env.step(None)
                    continue

                observation, reward, termination, truncation, _ = env.last()

                # Select an action only if the agent is not done
                if not (termination or truncation):
                    # Action is currently only selected based on mode == cooperative
                    action = select_action(agents[agent], observation, cooperative=cooperative, other_agents=agents.values())

                    # Wieso nicht analog zur evaluate() Methode?
                    #with torch.no_grad():
                    #    if cooperative:
                    #        action = agents[agent].act(obs_tensor)
                    #    else:
                    #        action = torch.argmax(agents[agent].model(obs_tensor)).item()
                else:
                    action = None  # Set action to None if the agent is done

                # Step in the environment
                env.step(action)
                next_observation, reward, termination, truncation, _ = env.last()

                # Update rewards
                total_rewards[agent] += reward

                # Log the step data
                data_records.append({
                    'Episode': episode + 1,
                    #'Step': step,
                    'Agent': agent,
                    'Mode': 'cooperative' if cooperative else 'individual',
                    'Action': action,
                    'Observation': observation,
                    'Next Observation': next_observation,
                    'Reward': reward,
                    'Total Reward': total_rewards[agent],
                    'Cooperative Reward': sum(total_rewards.values()) if cooperative else 0,
                    'Done': termination or truncation,
                })

                # Mark agent as done if termination or truncation occurred
                if termination or truncation:
                    done[agent] = True

                # Store experience for the agent (for training)
                agents[agent].remember(observation, action, reward, next_observation, termination or truncation)

                # Update agent if enough experience is collected
                if len(agents[agent].memory) >= 64:
                    agents[agent].update(64)

        env.close()

        # End of episode logging
        if cooperative:
            shared_reward = sum(total_rewards.values())
            total_cooperative_rewards.append(shared_reward)
            print(f"Episode {episode + 1}/{num_episodes} | Total Cooperative Reward: {total_rewards}")
        else:
            for agent in total_rewards:
                rewards_history[agent].append(total_rewards[agent])
            print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")

        # Save the recorded data to a CSV
        df_eval = pd.DataFrame(data_records)

        if not os.path.exists('evaluation_data'):
            os.makedirs('evaluation_data')

        df_eval.to_csv('evaluation_data/training_data.csv', index=False)

    print(f"Training data saved to evaluation_data/training_data.csv")

    # Calculate and return average rewards
    if cooperative:
        avg_rewards = sum(total_cooperative_rewards) / len(total_cooperative_rewards)
    else:
        avg_rewards = {agent: sum(rewards) / len(rewards) for agent, rewards in rewards_history.items()}

    return avg_rewards