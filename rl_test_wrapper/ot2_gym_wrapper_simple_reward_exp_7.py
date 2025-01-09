import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation
from stable_baselines3.common.callbacks import BaseCallback


class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()        
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=render)

        # Define action and observation space
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(3,), 
            dtype=np.float32
        )

        # Define the lower and upper bounds for the observation space
        low = -np.inf * np.ones(6, dtype=np.float32)
        high = np.inf * np.ones(6, dtype=np.float32)

        # Define the shape and dtype
        self.observation_space = spaces.Box(low=low, high=high, shape=(6,), dtype=np.float32)

        # Keep track of the number of steps
        self.steps = 0

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Define bounds for random position within the envelope
        low = np.array([-0.187, -0.1705, 0.1687], dtype=np.float32)
        high = np.array([0.253, 0.2195, 0.2896], dtype=np.float32)

        # Set initial position and goal position randomly within the bounds
        self.initial_position = np.random.uniform(low, high, size=(3,))
        self.goal_position = np.random.uniform(low, high, size=(3,))

        # Reset the state of the simulation environment
        sim_observation = self.sim.reset(num_agents=1)

        # Access the pipette position for the first robot ID
        robot_id = self.sim.robotIds[0]  # Get the first robot ID
        try:
            pipette_position = np.array(sim_observation[f'robotId_{robot_id}']['pipette_position'], dtype=np.float32)

            # Concatenate pipette position with the goal position for the observation
            observation = np.concatenate([pipette_position, self.goal_position], axis=0)

        except KeyError as e:
            print(f"KeyError: {e} - Check the sim_observation structure.")
            observation = np.zeros(6, dtype=np.float32)  # Fallback to a zeroed observation

        # Ensure the observation is the correct dtype (float32)
        observation = observation.astype(np.float32)

        # Reset additional environment variables
        self.steps = 0
        self.cumulative_reward = 0  # Initialize cumulative reward for the entire episode
        self.previous_distance = np.linalg.norm(self.initial_position - self.goal_position)  # Initialize distance

        # Set the start position in the simulation
        x, y, z = self.initial_position
        self.sim.set_start_position(x, y, z)

        # Include any additional info if needed
        info = {}

        return observation, info

    def step(self, action):
        # Ensure the action is properly formatted
        action = np.array(action, dtype=np.float32)

        # Append 0 for the drop action since we are only controlling the pipette position
        action = np.append(action, 0).astype(np.float32)

        # Call the simulation step function with the action
        sim_observation = self.sim.run([action])  # Actions are passed as a list

        # Access the pipette position for the first robot ID
        robot_id = self.sim.robotIds[0]
        try:
            pipette_position = np.array(sim_observation[f'robotId_{robot_id}']['pipette_position'], dtype=np.float32)

            # Construct the observation by concatenating pipette position and goal position
            observation = np.concatenate([pipette_position, self.goal_position], axis=0)
        except KeyError as e:
            print(f"KeyError: {e} - Check the sim_observation structure.")
            observation = np.zeros(6, dtype=np.float32)  # Fallback to a zeroed observation

        # Ensure the observation is the correct dtype (float32)
        observation = observation.astype(np.float32)

        # Calculate the distance between the pipette position and the goal position
        current_distance = np.linalg.norm(pipette_position - self.goal_position)

        # Step-specific reward
        reward = -1  # Penalize each step by -1
        if current_distance < self.previous_distance:
            reward += 5  # Reward for moving closer
        else:
            reward -= 10  # Penalty for moving further away

        # Update the previous distance
        self.previous_distance = current_distance

        # Give 100 points if within 001 distance to the goal
        if current_distance < 0.01:
            reward += 2000
            terminated = True
        else:
            terminated = False

        # Penalize if steps exceed max steps
        if self.steps >= self.max_steps:
            reward -= 100  # Penalty for exceeding max steps
            terminated = True
            truncated = True
        else:
            truncated = False

        # Update cumulative reward
        self.cumulative_reward += reward

        # Increment the step counter
        self.steps += 1

        # Return the updated observation, reward, and episode state
        info = {"cumulative_reward": self.cumulative_reward}  # Include cumulative reward in info
        return observation, float(reward), terminated, truncated, info


    def render(self, mode='human'):
        pass
    
    def close(self):
        self.sim.close()
