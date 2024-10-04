import subprocess
from random import randint


def main():
    for i in range(10):
        seed = randint(1, 10000)
        subprocess.run(["sbatch", "run_experiment.sh", "mainDQN.py", "--seed", str(seed)])
        subprocess.run(["sbatch", "run_experiment.sh", "mainMADDPG.py", "--seed", str(seed)])


if __name__ == '__main__':
    main()