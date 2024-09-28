# Shiny-MAS

## Overview

**Shiny-MAS** is a Multi-Agent System (MAS) framework designed to implement, train, and evaluate individual versus cooperative agent behaviour in a predator-prey environment using DQN and MADDPG learning approaches. This project leverages the PettingZoo library to create and manage multi-agent environments, with a focus on deep reinforcement learning (DRL) techniques.

## Project Structure

```
shiny-MAS-main/
│
├── mainDQN.py                      # Entry point of the application for deep Q-learning
├── mainMADDPG.py                   # Entry point of the application for Multi-Agent Deep Deterministic Policy Gradient
├── requirements.txt             # Python dependencies for the project
├── .gitignore                   
├── .simple_tag.py               # A copy of the modified version of the PettingZoo environment                    
│
├── .idea/                       # Configuration files for JetBrains IDEs (optional, created automatically)
│   ├── .gitignore
│   ├── misc.xml
│   ├── modules.xml
│   ├── project.iml
│   ├── vcs.xml
│   └── inspectionProfiles/
│       └── profiles_settings.xml
│
├── algorithms/                # Implementation of online-sourced DRL algorithms (version 01, via PyTorch)
│   ├── dqn.py                   # Deep Q-Network
│   ├── maddpg.py                # Multi-Agent Deep Deterministic Policy Gradient
│
└── utils/                       # Utility scripts for environment creation as well as training and evaluation
    ├── pettingzoo_env.py        # Environment initialization using PettingZoo
    ├── trainingDQN.py           # Training and evaluation loops for DQN
    └── trainingMADDPG.py        # Training and evaluation loops for MADDPG
```

## Getting Started

### Prerequisites

- Python 3.7+
- Virtual environment recommended (e.g., `venv` or `conda`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/shiny-MAS.git
   cd shiny-MAS-main
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

To run the main application, execute the `mainDQN.py` or `mainMADDPG.py` file:

```bash
python mainDQN.py [--num_envs <desired number of environments>]
python mainMADDPG.py [--num_envs <desired number of environments>]
```

This will start the training or evaluation process as configured.

### Directory Details

- **`algorithms/`**: This directory contains the implementations of DRL algorithms. These implementations are used to train and evaluate agents with different behaviour in pre-made environments. 

- **`utils/`**: Utility scripts necessary with environment initialization as well as model training and evaluation.

### Bibliography/Credits

1.	DQN (Deep Q-Learning Network):
o	Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.
2.	MADDPG (Multi-Agent Deep Deterministic Policy Gradient):
o	Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments. arXiv preprint arXiv:1706.02275.
o	MADDPG PyTorch Documentation: https://pytorch.org/rl/stable/tutorials/multiagent_competitive_ddpg.html
3.  tbd
