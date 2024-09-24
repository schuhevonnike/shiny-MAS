#import os
#import pandas as pd
import torch
from environments.pettingzoo_env import make_env

def select_action(agent, observation):
    with torch.no_grad:
        action = agent.act(observation)
    return action

def train(agents, num_episodes):
    env = make_env()
    rewards_history = {agent: [] for agent in agents}
    #data_records = []

    for episode in range(num_episodes):
        env.reset()
        # env.reset(seed=42)
        total_rewards = {agent: 0 for agent in env.possible_agents}
        # Initialize done flag for tracking done state for each agent.
        done = {agent: False for agent in env.possible_agents}
        # Continue iterating until all agents are done.
        while not all(done.values()):
            for agent in env.agent_iter():
                # Check if the agent is done. In that case, skip to next agent.
                if done[agent]:
                    env.step(None)
                    continue
                # Call the last() method to gather the relevant data from the last step taken in the environment.
                # In the first iteration, the last step equals the initialization of the environment itself.
                # Deprecate any remaining info (', _') .
                # Observation = input, reward = target
                observation, reward, termination, truncation, _ = env.last()
                # Save observation and combine with next_observation.
                # Select an action only if the agent is not done.
                if not (termination or truncation):
                    # Converts the dict of observations into a torch tensor. The unsqueeze() method adds a new dimension at index 0 which is suiting for batch processing.
                    obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        action = torch.argmax(agents[agent].model(obs_tensor)).item()
                else:
                    action = None  # Set action to None if the agent is marked as done.
                # Update rewards after each action taken.
                total_rewards[agent] += reward
                # Step in the environment with the selected action.
                # This instance skips to the next agent.
                env.step(action)
                # This next line is still unresolved:
                next_observation, reward, termination, truncation, _ = env.last()  # .last() refers to the current agent, in the case of adv_0 taking a step the next agent would then be adv_1
                # However, we would need the next_observation of adv_0 instead.
                #next_obs_tensor = torch.tensor(next_observation, dtype=torch.float32).unsqueeze(0)
                '''
                # Log the step data
                data_records.append({
                    'Episode': episode + 1,
                    'Agent': agent,
                    'Action': action,
                    'Observation': observation,
                    'Reward': reward,
                    'Total Reward': total_rewards[agent],
                    'Done': termination or truncation,
                })
                '''
                # Mark agent as done when termination or truncation occur.
                if termination or truncation:
                    done[agent] = True
                # Store experience for each agent during training.
                agents[agent].remember(observation, action, reward, next_observation, termination or truncation)
                # Update agent if enough experience is collected.
                if len(agents[agent].memory) >= 64:
                    agents[agent].update(64)
        # Close the environment when all agents are marked as done.
        env.close()
        # Logging rewards at the end of each episode.
        for agent in total_rewards:
            rewards_history[agent].append(total_rewards[agent])
        print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")
        '''
        # Save the recorded data to a CSV
        df_eval = pd.DataFrame(data_records)

        if not os.path.exists('evaluation_data'):
            os.makedirs('evaluation_data')

        df_eval.to_csv('evaluation_data/training_data.csv', index=False)
        '''
    #print(f"Training data saved to evaluation_data/training_data.csv")
    # Calculate and return average rewards for each training episode.
    avg_rewards = {agent: sum(rewards) / len(rewards) for agent, rewards in rewards_history.items()}
    return avg_rewards