import os
import time
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
        episode_start_time = time.time()
        env.reset()
        total_rewards = {agent: 0 for agent in env.possible_agents}
        done = False
        step = 0  # Initialize step counter for each episode

        while not done:
            step += 1
            for agent in env.possible_agents:
            #for agent in env.agent_iter():
                observation, reward, termination, truncation, _ = env.last()

            if termination or truncation:
                done = True
                env.step(None)
                break


            #obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)


            action = select_action(agents[agent], observation, cooperative=cooperative, other_agents=agents.values())
            env.step(action)
            next_observation, reward, termination, truncation, _ = env.last()

            # Update rewards
            total_rewards[agent] += reward

            # Store step data in the list
            data_records.append({
                'Episode': episode + 1,
                'Step': step,
                'Agent': agent,
                'Mode': 'cooperative' if cooperative else 'individual',
                'Action': action,
                'Observation': observation,
                'Next Observation': next_observation,
                'Reward': reward,
                'Total Reward': total_rewards[agent],
                'Cooperative Reward': sum(total_rewards.values()) if cooperative else 0,
                'Done': termination or truncation,
                'Step Duration': time.time() - episode_start_time
            })

            # Store experience and update agent model
            agents[agent].remember(observation, action, reward, next_observation, termination or truncation)

            # Update agent if it has enough experience (for agents with replay memory)
            if len(agents[agent].memory) >= 64:  # Using batch size 64 for example
                agents[agent].update(64)
        env.close()

        # End of episode logging
        if cooperative:
            shared_reward = sum(total_rewards.values())
            total_cooperative_rewards.append(shared_reward)
            print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")
            #print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {shared_reward}")
        else:
            for agent in total_rewards:
                rewards_history[agent].append(total_rewards[agent])
            print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")

    # Convert records to a DataFrame and save to CSV
    df_eval = pd.DataFrame(data_records)

    # Ensure a directory for saving exists
    if not os.path.exists('evaluation_data'):
        os.makedirs('evaluation_data')

    # Save to CSV
    df_eval.to_csv('evaluation_data/training_data.csv', index=False)
    print(f"Data saved to evaluation_data/training_data.csv")

    # Calculate average rewards for each agent or cooperative rewards
    if cooperative:
        avg_rewards = sum(total_cooperative_rewards) / len(total_cooperative_rewards)
    else:
        avg_rewards = {agent: sum(rewards) / len(rewards) for agent, rewards in rewards_history.items()}

    return avg_rewards