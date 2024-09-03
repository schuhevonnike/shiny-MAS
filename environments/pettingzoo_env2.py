from pettingzoo.mpe import simple_tag_v3
from pettingzoo.utils import wrappers

def make_env():
    # Load the simple_tag_v3 environment as ParallelEnv
    env = simple_tag_v3.env()

    # Apply wrappers that are compatible with AEC environments
    env = wrappers.OrderEnforcingWrapper(env)
    return env

if __name__ == "__main__":
    env = make_env()

    # Initialize the environment and get initial observations
    env.reset()
    print(f"Environment initialized")  # Debugging print to verify output

    #agent_iter() is a method used to iterate over all possible agents
    for agent in env.agent_iter():
        #Retrieve obs, rew and further info from the previous agent/step
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample()  # Take random actions

        # Execute the action
        env.step(action)  # Do not unpack, as step does not return anything

        if termination or truncation:
            env.reset()  # Reset the environment for the next episode

    env.render()  # Adjust render mode as needed
    env.close()
