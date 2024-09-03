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
    print("Environment initialized")

    # Dictionary to store the last observation for each agent in the iteration
    last_observations = {}

    # Main interaction loop, this now includes communication between agents via communication of the recent observation
    for agent in env.agent_iter():
        # Retrieve the recent observation, reward, termination, truncation, and info for the current agent
        observation, reward, termination, truncation, info = env.last()
        last_observations[agent] = observation  # Store the last observation
        print(f"Last observations' shape: {last_observations.shape}")
        if termination or truncation:
            action = None  # No action if the agent is done
        else:
            action = env.action_space(agent).sample()  # Sample a random action

        env.step(action)  # Execute the action

        # Reset environment if any agent's episode has ended
        if termination or truncation:
            env.reset()

        # Debugging prints to verify interactions
        print(f"Agent: {agent}, Action: {action}, Observation: {observation}, Reward: {reward}, Done: {termination or truncation}")

    env.render()  # Render the environment
    env.close()