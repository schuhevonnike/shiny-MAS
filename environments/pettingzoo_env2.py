from pettingzoo.mpe import simple_tag_v3
from pettingzoo.utils import wrappers
from pettingzoo.utils.env import AECEnv, ParallelEnv
from pettingzoo.utils.conversions import parallel_to_aec, aec_to_parallel, aec_to_parallel_wrapper

def parallel_env():
    # Load the simple_tag_v3 environment as ParallelEnv
    env = simple_tag_v3.parallel_env()

    # Convert to AEC environment for applying AEC-specific wrappers
    if isinstance(env, ParallelEnv):
        env = parallel_to_aec(env)  # Convert to AEC environment

    # Apply wrappers that are compatible with AEC environments
    env = wrappers.OrderEnforcingWrapper(env)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-10)  # Example wrapper

    # Convert back to ParallelEnv if necessary
    if isinstance(env, AECEnv):
        env = aec_to_parallel(env)

    return env

if __name__ == "__main__":
    env = parallel_env()

    # Initialize the environment and get initial observations
    observations = env.reset()
    print(f"Initial Observations: {observations}")  # Debugging print to verify output

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample()  # Take random actions

        # Execute the action
        observations, reward, done, info = env.step(action)

        if termination or truncation:
            observations = env.reset()  # Reset the environment for the next episode

    env.render()  # Adjust render mode as needed
    env.close()
