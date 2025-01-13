import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from stable_baselines3.common.env_checker import check_env
from ot2_gym_wrapper_simple_reward_exp_8 import OT2Env
from stable_baselines3 import PPO
import time
import wandb
from wandb.integration.sb3 import WandbCallback
import argparse
import typing_extensions
from clearml import Task
from stable_baselines3.common.callbacks import BaseCallback
import torch  

# Teun van der Wolf s235874

# Replace Pendulum-v1/YourName with your own project name (Folder/YourName, e.g. 2022-Y2B-RoboSuite/Michael)
task = Task.init(project_name='Mentor Group A/Group 1/Teun', # NB: Replace YourName with your own name
                    task_name='Experiment1')

# Copy these lines exactly as they are
# Setting the base docker image
task.set_base_docker('deanis/2023y2b-rl:latest')
# Setting the task to run remotely on the default queue
task.execute_remotely(queue_name="default")

# Argument parsing for hyperparameters
parameter_parser = argparse.ArgumentParser()
parameter_parser.add_argument("--learning_rate", type=float, default=1e-5) # Lower learning rate
parameter_parser.add_argument("--batch_size", type=int, default=64) # Batch size
parameter_parser.add_argument("--n_steps", type=int, default=512) # number of steps
parameter_parser.add_argument("--n_epochs", type=int, default=10) # number of epochs
parameter_parser.add_argument("--discount_factor", type=float, default=0.98)  # Gamma value
parameter_parser.add_argument("--vf_coef", type=float, default=0.5)  # Value function coefficient
parameter_parser.add_argument("--clip_threshold", type=float, default=0.2)  # Clipping range
parameter_parser.add_argument("--policy_type", type=str, default="MlpPolicy")  # Policy type
arguments, unknown_args = parameter_parser.parse_known_args()

args = parameter_parser.parse_args()

os.environ['WANDB_API_KEY'] = '17b671297e98466f9af4baa04230fcd84aec26c3'

# Initialize WandB project
run = wandb.init(project="rl_batch_size", sync_tensorboard=True)

# Log the hyperparameters to WandB
wandb.config.update(vars(arguments))  # This will log the arguments to the WandB config

# Custom environment initialization (ensure OT2Env adheres to the Gym API)
try:
    training_env = OT2Env(render=False) # Render disabled for training efficiency
except Exception as e:
    raise RuntimeError(f"Failed to initialize the training environment: {e}")
  

# PPO model initialization with parsed parameters
ppo_model = PPO(
    policy=arguments.policy_type,
    env=training_env,
    verbose=1,
    learning_rate=arguments.learning_rate,
    batch_size=arguments.batch_size,
    n_steps=arguments.n_steps,
    n_epochs=arguments.n_epochs,
    gamma=arguments.discount_factor,
    ent_coef=arguments.vf_coef,
    clip_range=arguments.clip_threshold,
    tensorboard_log=f"tensorboard_logs/{run.id}",
)

# Ensure directories for saving models are prepared
model_storage_path = f"saved_models/{run.id}"
os.makedirs(model_storage_path, exist_ok=True)

# Define a callback for tracking the best model and saving at intervals
class EnhancedModelSaver(BaseCallback):
    '''A callback to save intermediate and best-performing models during training.'''

    def __init__(self, save_directory, save_interval=50000, log_detail=0):
        '''
        Initialize the EnhancedModelSaver.

        Args:
            save_directory (str): Directory to save models.
            save_interval (int): Save model every `save_interval` steps.
            log_detail (int): Verbosity level.
        '''
        super().__init__(log_detail)
        self.save_directory = save_directory
        self.save_interval = save_interval
        self.highest_reward = -float("inf")  # Start with a low benchmark

    def _on_step(self) -> bool:
        '''
        Called at each step of training.

        Saves models periodically and updates the best model based on reward.

        Returns:
            bool: Whether to continue training.
        '''
        if self.n_calls % self.save_interval == 0:
            periodic_model_path = os.path.join(self.save_directory, f"model_step_{self.num_timesteps}.zip")
            self.model.save(periodic_model_path)
            wandb.save(periodic_model_path)
            print(f"Saved intermediate model at {self.num_timesteps} steps")

        # Evaluate episode rewards from the `info` dictionary
        episode_stats = self.locals["infos"][0].get("episode", {})
        if "r" in episode_stats:
            episode_reward = episode_stats["r"]
            if episode_reward > self.highest_reward:
                self.highest_reward = episode_reward
                best_model_path = os.path.join(self.save_directory, "best_model.zip")
                self.model.save(best_model_path)
                wandb.save(best_model_path)
                print(f"New best model saved with reward {self.highest_reward} at {self.num_timesteps} steps")

        return True

# Define a callback for logging training metrics
class MetricsLogger(BaseCallback):
    '''A callback to log episode rewards, lengths, and success rates during training.'''

    def __init__(self, log_detail=0):
        '''
        Initialize the MetricsLogger.

        Args:
            log_detail (int): Verbosity level.
        '''
        super().__init__(log_detail)
        self.logged_rewards = []
        self.logged_lengths = []
        self.success_metrics = []

    def _on_step(self) -> bool:
        '''
        Called at each step of training.

        Logs episode rewards, lengths, and success rates to WandB.

        Returns:
            bool: Whether to continue training.
        '''
        # Track episode-specific details 
        if "episode" in self.locals["infos"][0]:
            episode_data = self.locals["infos"][0].get("episode", {})
            if "r" in episode_data:
                episode_reward = episode_data["r"]
                self.logged_rewards.append(episode_reward)
                wandb.log({"episode_reward": episode_reward}, step=self.num_timesteps)
                print(f"Logged episode reward: {episode_reward}")

            if "l" in episode_data:
                episode_length = episode_data["l"]
                self.logged_lengths.append(episode_length)
                wandb.log({"episode_length": episode_length}, step=self.num_timesteps)

        # Optional success metric tracking
        success_stat = self.locals["infos"][0].get("success", None)
        if success_stat is not None:
            self.success_metrics.append(success_stat)
            wandb.log({"success_rate": np.mean(self.success_metrics)}, step=self.num_timesteps)

        return True

# Instantiate and combine callbacks
periodic_model_saver = EnhancedModelSaver(save_directory=model_storage_path, save_interval=100000, log_detail=1)
training_metrics_logger = MetricsLogger()
wandb_callback_handler = WandbCallback(
    model_save_freq=2000,
    model_save_path=model_storage_path,
    verbose=2,
)

# Run the PPO model training with gradient clipping
print("Starting training process")
total_training_steps = 2000000  # Modify as needed for experimentation

# Adding gradient clipping to PPO during training
for step in range(total_training_steps):
    ppo_model.learn(
        total_timesteps=1,  # Single step
        callback=[wandb_callback_handler, training_metrics_logger, periodic_model_saver],
        reset_num_timesteps=False,
    )
    # Clip gradients after each update
    for param in ppo_model.policy.parameters():
        if param.grad is not None:
            torch.nn.utils.clip_grad_norm_(param, max_norm=1.0)

print("Training completed")

# Save the final iteration model
final_model_path = f"{model_storage_path}/final_model.zip"
ppo_model.save(final_model_path)
wandb.save(final_model_path)
print(f"Final model saved to {final_model_path}")