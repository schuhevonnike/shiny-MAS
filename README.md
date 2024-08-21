# Shiny-MAS

## Overview

**Shiny-MAS** is a Multi-Agent System (MAS) framework designed to implement, train, and evaluate various reinforcement learning algorithms in simulated environments. This project leverages the PettingZoo library to create and manage multi-agent environments, with a focus on deep reinforcement learning (DRL) techniques.

## Project Structure

'''
shiny-MAS-main/
│
├── main.py                      # Entry point of the application
├── requirements.txt             # Python dependencies for the project
├── .gitignore                   # Git ignore file
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
├── algorithms01/                # Implementation of online-sourced DRL algorithms (version 01, via PyTorch)
│   ├── dqn.py                   # Deep Q-Network
│   ├── maddpg.py                # Multi-Agent Deep Deterministic Policy Gradient
│   ├── ppo.py                   # Proximal Policy Optimization
│   ├── sac.py                   # Soft Actor-Critic
│
├── algorithms02/                # Implementation of research-based DRL algorithms (version 02)
│   ├── dqn.py                   # Deep Q-Network
│   ├── maddpg.py                # Multi-Agent Deep Deterministic Policy Gradient
│   ├── ppo.py                   # Proximal Policy Optimization
│   ├── sac.py                   # Soft Actor-Critic
│
├── environments/                # Custom environments, such as simple_tag - a hunter-prey-scenario, built using PettingZoo
│   └── pettingzoo_env.py        # Environment wrapper for handling PettingZoo
│
└── utils/                       # Utility scripts for training, evaluation, and metrics
    ├── evaluation.py            # Evaluation functions for trained models
    ├── metrics.py               # Performance metrics for analysis, e.g. convergence rate, success rate, mean value and standard deviation of rewards
    └── training.py              # Training loops and functions
'''

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

To run the main application, execute the `main.py` file:

```bash
python main.py
```

This will start the training or evaluation process as configured.

### Directory Details

- **`algorithms01/` and `algorithms02/`**: These directories contain different implementations of DRL algorithms. These implementations are used to train agents in pre-made environments. In this study, they are subject of comparison.

- **`environments/`**: Contains environments built using the PettingZoo library, which is a fork of the Gymnasium library, compatible with multi-agent scenarios.

- **`utils/`**: Utility scripts necessary with model training, evaluation, and performance metrics.

### Bibliography/Credits

1.	DQN (Deep Q-Learning Network):
o	Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.
o	DQN PyTorch Implementation: https://github.com/karpathy/pg-py
2.	PPO (Proximal Policy Optimization):
o	Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347.
o	PPO PyTorch Implementation: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
3.	SAC (Soft Actor-Critic):
o	Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. arXiv preprint arXiv:1801.01290.
o SAC PyTorch Implementation: https://github.com/pranz24/pytorch-soft-actor-critic
4.	MADDPG (Multi-Agent Deep Deterministic Policy Gradient):
o	Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments. arXiv preprint arXiv:1706.02275.
o	MADDPG PyTorch Documentation: https://pytorch.org/rl/stable/tutorials/multiagent_competitive_ddpg.html
5. additional sources via OLAT 
