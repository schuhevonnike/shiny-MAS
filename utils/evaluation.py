import torch
import pandas as pd
import os
from environments.pettingzoo_env import (make_env)

def evaluate(agents, num_episodes):
    env = make_env()
    rewards_history = {agent: [] for agent in agents}
    #data_records = []

    for episode in range(num_episodes):
        env.reset()
