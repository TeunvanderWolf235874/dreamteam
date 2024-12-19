from stable_baselines3.common.env_checker import check_env
from ot2_gym_wrapper import OT2Env
from stable_baselines3 import PPO
import time
import wandb
from wandb.integration.sb3 import WandbCallback
import os
import argparse
import typing_extensions
from clearml import Task

# Replace Pendulum-v1/YourName with your own project name (Folder/YourName, e.g. 2022-Y2B-RoboSuite/Michael)
task = Task.init(project_name='Mentor Group A/Group 1/Teun', # NB: Replace YourName with your own name
                    task_name='Experiment1')

# Copy these lines exactly as they are
# Setting the base docker image
task.set_base_docker('deanis/2023y2b-rl:latest')
# Setting the task to run remotely on the default queue
task.execute_remotely(queue_name="default")

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)

args = parser.parse_args()

os.environ['WANDB_API_KEY'] = '17b671297e98466f9af4baa04230fcd84aec26c3'

# Initialize WandB project
run = wandb.init(project="rl_test_wrapper", sync_tensorboard=True)

# Instantiate your custom environment
wrapped_env = OT2Env()  # Modify this to match your wrapper class

# Assuming 'wrapped_env' is your wrapped environment instance
check_env(wrapped_env)

# Load your custom environment
env = wrapped_env

# Add tensorboard logging to the model
model = PPO('MlpPolicy', env, verbose=1, 
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size, 
            n_steps=args.n_steps, 
            n_epochs=args.n_epochs, 
            tensorboard_log=f"runs/{run.id}")

# Create WandB callback
wandb_callback = WandbCallback(
    model_save_freq=10000,  # Adjust how often to save during training (optional)
    model_save_path=f"models/{run.id}",
    verbose=2,
)

# Variable for how often to save the model
timesteps = 100000

# Initialize variables for tracking the best model
best_reward = -float('inf')  # Start with a very low reward
best_model_path = f"models/{run.id}/best_model"  # Path to save the best model

# Training loop
for i in range(10):
    # Add the reset_num_timesteps=False argument to prevent the model from resetting the timestep counter
    # Add the tb_log_name argument to log TensorBoard data to the correct folder
    model.learn(total_timesteps=timesteps, callback=wandb_callback, progress_bar=True, reset_num_timesteps=False, tb_log_name=f"runs/{run.id}")

    # Compute and log the mean reward (or another performance metric)
    # Here we log the mean reward across the training period (you can customize it as needed)
    mean_reward = model.get_reward()  # Placeholder for actual reward logic
    wandb.log({'mean_reward': mean_reward})  # Log to wandb

    # Check if the model has improved (higher reward)
    if mean_reward > best_reward:
        best_reward = mean_reward
        model.save(best_model_path)  # Save the best model

    # Save the model periodically to the models folder with the run id and the current timestep
    model.save(f"models/{run.id}/{timesteps * (i + 1)}")
