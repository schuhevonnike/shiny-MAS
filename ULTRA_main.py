import multiprocessing
import torch
from mainMADDPG import start_seed as MADDPG_start_seed
from mainDQN import start_seed as DQN_start_seed
# Function to check if a GPU with CUDA is available
def is_cuda_available():
    return torch.cuda.is_available()

# Determine the number of available CPU cores
cpu_count = multiprocessing.cpu_count()

# Use the maximum number of threads possible based on the CPU count
def run_experiments_in_parallel(run_experiment, seeds):
    # Using multiprocessing pool to parallelize the process across available CPU cores
    with multiprocessing.Pool(processes=cpu_count) as pool:
        pool.map(run_experiment, seeds)

# Modify the train and evaluate methods to use GPU if available
def run_experiment_with_cuda_support(algorithm_name, seed):
    if is_cuda_available():
        device = torch.device("cuda")
        print(f"Running {algorithm_name} with seed {seed} on GPU using CUDA")
    else:
        device = torch.device("cpu")
        print(f"Running {algorithm_name} with seed {seed} on CPU")

    # Assuming train() and evaluate() use torch and can work with the device




# Example usage
if __name__ == "__main__":
    # Define the list of seeds
    seeds = [i for i in range(12)]

    # Run experiments in parallel
    run_experiments_in_parallel(lambda seed: run_experiment_with_cuda_support("MADDPG", seed), seeds)
