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
        # They must be gym.spaces objects

        # Action space has to be -1 till 1
        # Observational space can be any value in the 6 value array
        # lr 0.001 and go up and down
        # Freeze parameters when they reach an optimal value    
        # Reward function is a hyper parameter: how fast does it reach the optimal point. 
        # bREADCRUM reawrd. olnly get a reward when you actively move into the directions you have to be moving

      
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(3,), 
            dtype=np.float32
        )

        # Define the lower and upper bounds for the observation space
        low = -np.inf * np.ones(6, dtype=np.float32)  # [-inf, -inf, -inf, -inf, -inf, -inf]
        high = np.inf * np.ones(6, dtype=np.float32)  # [ inf,  inf,  inf,  inf,  inf,  inf]

        # Define the shape and dtype
        shape = (6,)  # 6 values: pipette_x, pipette_y, pipette_z, goal_x, goal_y, goal_z
        dtype = np.float32
        self.observation_space = spaces.Box(low=low, high=high, shape=shape, dtype=dtype)

        # keep track of the number of steps
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
        self.min_error = float('inf')  # Minimum error starts at infinity
        self.previous_distance = np.linalg.norm(self.initial_position - self.goal_position) # Initialize distance
 
        # Set the start position in the simulation
        # Set the start position in the simulation (pass x, y, z separately)
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

        # Reward system
        reward = 0
        if hasattr(self, 'previous_distance'):
            # Give 10 points if the agent gets closer to the goal
            if current_distance < self.previous_distance:
                reward += 10
            else:
                reward -= 1  # Small penalty for not improving

        # Give 100 points if within 0.01 distance to the goal
        if current_distance < 0.01:
            reward += 100
            terminated = True
        else:
            terminated = False

        # Penalize if steps exceed max steps
        if self.steps >= self.max_steps:
            reward -= 10  # Penalty for exceeding max steps
            terminated = True
            truncated = True
        else:
            truncated = False

        # Update the previous distance for the next step
        self.previous_distance = current_distance

        # Increment the step counter
        self.steps += 1

        # Return the updated observation, reward, and episode state
        info = {}
        return observation, float(reward), terminated, truncated, info

    def render(self, mode='human'):
        pass
    
    def close(self):
        self.sim.close()

