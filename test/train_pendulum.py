from stable_baselines3 import PPO
import gymnasium as gym
import time

import wandb
from wandb.integration.sb3 import WandbCallback
import os

import argparse

from clearml import Task

# Replace Pendulum-v1/YourName with your own project name (Folder/YourName, e.g. 2022-Y2B-RoboSuite/Michael)
task = Task.init(project_name='Pendulum-v1/235874_Teun', # NB: Replace YourName with your own name
                    task_name='Experiment1')

#copy these lines exactly as they are
#setting the base docker image
task.set_base_docker('deanis/2023y2b-rl:latest')
#setting the task to run remotely on the default queue
task.execute_remotely(queue_name="default")

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)

args = parser.parse_args()

os.environ['WANDB_API_KEY'] = '17b671297e98466f9af4baa04230fcd84aec26c3'

# initialize wandb project
run = wandb.init(project="sb3_pendulum_demo",sync_tensorboard=True)

# Set up the environment and model
env = gym.make('Pendulum-v1', g=9.81)

# add tensorboard logging to the model
model = PPO('MlpPolicy', env, verbose=1, 
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size, 
            n_steps=args.n_steps, 
            n_epochs=args.n_epochs, 
            tensorboard_log=f"runs/{run.id}",)


# create wandb callback
# Create Wandb callback
wandb_callback = WandbCallback(
    model_save_freq=1000,
    model_save_path=f"models/{run.id}",
    verbose=2,
)


# variable for how often to save the model
timesteps = 100000
for i in range(10):
    # add the reset_num_timesteps=False argument to the learn function to prevent the model from resetting the timestep counter
    # add the tb_log_name argument to the learn function to log the tensorboard data to the correct folder
    model.learn(total_timesteps=timesteps, callback=wandb_callback, progress_bar=True, reset_num_timesteps=False,tb_log_name=f"runs/{run.id}")
    # save the model to the models folder with the run id and the current timestep
    model.save(f"models/{run.id}/{timesteps*(i+1)}")



