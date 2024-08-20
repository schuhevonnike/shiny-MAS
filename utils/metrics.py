import numpy as np

def calculate_metrics(rewards):
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    return mean_reward, std_reward
