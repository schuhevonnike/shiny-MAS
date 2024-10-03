import os
import random
import numpy as np
import pandas as pd
import torch
from utils.pettingzoo_env import make_env

def train(agents, num_episodes, seed, device):
    env = make_env()
    rewards_history = {agent: [] for agent in agents}
    data_records = []

    for episode in range(num_episodes):
        env.reset()
        total_rewards = {agent: 0 for agent in env.possible_agents}
        done = {agent: False for agent in env.possible_agents}
        last_observation = {agent: None for agent in env.possible_agents}
        last_action = {agent: None for agent in env.possible_agents}

        for agent in env.agent_iter():
            observation, reward, termination, truncation, _ = env.last()
            current_done = termination or truncation
            done[agent] = current_done

            # Verschieben der Beobachtung auf das Gerät
            observation = torch.tensor(observation, dtype=torch.float32).to(device)

            # Update total rewards
            total_rewards[agent] += reward

            # Wenn es nicht der erste Zug des Agenten ist, speichere die Transition
            if last_observation[agent] is not None and last_action[agent] is not None:
                agents[agent].remember(
                    state=last_observation[agent],
                    action=last_action[agent],
                    reward=torch.tensor([reward], dtype=torch.float32).to(device),
                    next_state=observation,
                    done=current_done
                )

            # Aktion auswählen
            if not current_done:
                # Epsilon-greedy Implementierung
                if random.random() > agents[agent].epsilon:
                    with torch.no_grad():
                        obs_tensor = observation.unsqueeze(0)
                        q_values = agents[agent].model(obs_tensor)
                        action = torch.argmax(q_values).item()
                else:
                    action = env.action_space(agent).sample()
            else:
                action = None

            last_observation[agent] = observation
            last_action[agent] = action

            # Loggen der Schritt-Daten
            data_records.append({
                'Episode': episode,
                'Agent': agent,
                'Action': action,
                'Observation': observation.cpu().numpy(),
                'Reward': reward,
                'Done': current_done,
            })

            # Schritt in der Umgebung ausführen
            env.step(action)

        # Agenten updaten
        for agent in agents:
            if len(agents[agent].memory) >= 256:
                for i in range(100):
                    agents[agent].update(256, device)

        # Belohnungen am Ende jeder Episode loggen
        for agent in total_rewards:
            rewards_history[agent].append(total_rewards[agent])
        print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")
        print(f"Mean Reward last 100 Episodes:")
        for agent in agents:
            mean_reward = np.mean(rewards_history[agent][-100:])
            print(f"Agent {agent} Reward: {mean_reward}")
        print()

    # Daten als CSV speichern
    df_eval = pd.DataFrame(data_records)

    if not os.path.exists('data_exportDQN'):
        os.makedirs('data_exportDQN')

    df_eval.to_csv(f'data_exportDQN/training_data_{seed}.csv', index=False)
    print(f"\nTraining data saved to data_exportDQN/training_data_{seed}.csv")

    avg_rewards = {agent: np.mean(rewards) for agent, rewards in rewards_history.items()}
    return avg_rewards


def evaluate(agents, num_episodes, seed, device):
    env = make_env()
    rewards_history = {agent: [] for agent in agents}
    data_records = []

    for episode in range(num_episodes):
        env.reset()
        total_rewards = {agent: 0 for agent in env.possible_agents}
        done = {agent: False for agent in env.possible_agents}
        last_observation = {agent: None for agent in env.possible_agents}
        last_action = {agent: None for agent in env.possible_agents}

        for agent in env.agent_iter():
            observation, reward, termination, truncation, _ = env.last()
            current_done = termination or truncation
            done[agent] = current_done

            # Verschieben der Beobachtung auf das Gerät
            observation = torch.tensor(observation, dtype=torch.float32).to(device)

            # Update total rewards
            total_rewards[agent] += reward

            # Wenn es nicht der erste Zug des Agenten ist, speichere die Transition
            if last_observation[agent] is not None and last_action[agent] is not None:
                agents[agent].remember(
                    state=last_observation[agent],
                    action=last_action[agent],
                    reward=torch.tensor([reward], dtype=torch.float32).to(device),
                    next_state=observation,
                    done=current_done
                )

            # Aktion auswählen (keine Exploration während der Evaluation)
            if not current_done:
                with torch.no_grad():
                    obs_tensor = observation.unsqueeze(0)
                    q_values = agents[agent].model(obs_tensor)
                    action = torch.argmax(q_values).item()
            else:
                action = None

            last_observation[agent] = observation
            last_action[agent] = action

            # Loggen der Schritt-Daten
            data_records.append({
                'Episode': episode,
                'Agent': agent,
                'Action': action,
                'Observation': observation.cpu().numpy(),
                'Reward': reward,
                'Done': current_done,
            })

            # Schritt in der Umgebung ausführen
            env.step(action)

        # Belohnungen am Ende jeder Episode loggen
        for agent in total_rewards:
            rewards_history[agent].append(total_rewards[agent])
        print(f"Episode {episode + 1}/{num_episodes} | Total Rewards: {total_rewards}")
        print(f"Mean Reward last 100 Episodes:")
        for agent in agents:
            mean_reward = np.mean(rewards_history[agent][-100:])
            print(f"Agent {agent} Reward: {mean_reward}")
        print()

    # Daten als CSV speichern
    df_eval = pd.DataFrame(data_records)

    if not os.path.exists('data_exportDQN'):
        os.makedirs('data_exportDQN')

    df_eval.to_csv(f'data_exportDQN/evaluation_data_{seed}.csv', index=False)
    print(f"\nEvaluation data saved to data_exportDQN/evaluation_data_{seed}.csv")

    avg_rewards = {agent: np.mean(rewards) for agent, rewards in rewards_history.items()}
    return avg_rewards
