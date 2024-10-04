import multiprocessing
from mainDQN import start_seed as DQN_start_seed
from mainMADDPG import start_seed as MADDPG_start_seed

def run_algorithm(args):
    algorithm, seed = args
    if algorithm == 'DQN':
        DQN_start_seed(seed)
    elif algorithm == 'MADDPG':
        MADDPG_start_seed(seed)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

if __name__ == "__main__":
    #seeds = [i for i in range(12)]
    seeds = [42, 1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    tasks = []

    # Create tasks for both algorithms with each seed
    for seed in seeds:
        tasks.append(('DQN', seed))
        tasks.append(('MADDPG', seed))

    # Use multiprocessing to run tasks in parallel
    with multiprocessing.Pool() as pool:
        pool.map(run_algorithm, tasks)
