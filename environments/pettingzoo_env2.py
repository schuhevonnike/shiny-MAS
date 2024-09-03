from pettingzoo.mpe import simple_tag_v3
from pettingzoo.utils import wrappers


def make_env():
    # Load the simple_tag_v3 environment
    env = simple_tag_v3.env()
    # Apply wrappers compatible with AEC environments
    env = wrappers.OrderEnforcingWrapper(env)
    return env


if __name__ == "__main__":
    env = make_env()
    observation = env.reset()  # Initialize the environment
    print(f"Initial observation after reset: {observation}")

    # Dictionary to store the last observation for each agent in the iteration
    last_observations = {}

    # Main interaction loop
    done = False
    while not done:
        for agent in env.agent_iter():
            observation, reward, termination, truncation, _ = env.last()
            last_observations[agent] = observation  # Store the last observation
            print(f"Last observations' shape for {agent}: {observation.shape}")    # Print the shape of the last observation

            if termination or truncation:
                action = None
            else:
                action = env.action_space(agent).sample() # Sample a random action

            next_observation, reward, done, info = env.step(action) # Execute the action and store the according values
            #env.step(action)  #Option B: Only execute the action
            # Log the step result
            print(f"Step result: {next_observation}, {reward}, {done}, {info}")

            if termination or truncation:
            # Reset environment if any agent's episode has ended
                observation = env.reset()
                done = True
                break  # Exit agent iteration loop

    env.render()  # Render the environment
    env.close()
