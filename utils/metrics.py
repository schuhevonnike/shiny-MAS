import numpy as np

def calculate_metrics(adversary_rewards, cooperator_rewards):
    #Calculate mean and standard deviation of rewards for adversary and cooperator agents:
    adversary_mean_reward = np.mean(adversary_rewards) # float, mean of adversary rewards using the list of rewards for adversary agents
    adversary_std_reward = np.std(adversary_rewards) # float, std of adversary rewards
    cooperator_mean_reward = np.mean(cooperator_rewards) # float, mean of cooperator rewards using the list of rewards for cooperator agents
    cooperator_std_reward = np.std(cooperator_rewards) # float, std of cooperator rewards

    return adversary_mean_reward, adversary_std_reward, cooperator_mean_reward, cooperator_std_reward
