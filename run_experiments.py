import subprocess


def main():
    for seed in [222,351,398,453,545,580,752,827,841,983]:
        subprocess.run(["sbatch", "run_experiment.sh", "mainDQN.py", "--seed", str(seed)])
        subprocess.run(["sbatch", "run_experiment.sh", "mainMADDPG.py", "--seed", str(seed)])


if __name__ == '__main__':
    main()