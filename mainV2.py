import torch
import pandas
import os
import


# Experiment parameters
agent_types = ['adversary', 'cooperator']
# ...

# Cuda cores are better suited for such tasks, so ideally, we want to run the program on a cuda. If not possible, use the standard CPU.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_experiment(env):
    # Initialize separate dictionaries for adversary and cooperator agents
    adversary_agents = {}
    cooperator_agents = {}

