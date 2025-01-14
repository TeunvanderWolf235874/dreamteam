import optuna
from stable_baselines3.common.env_checker import check_env
from ot2_gym_wrapper_simple_reward_exp_9 import OT2Env
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
from clearml import Task
from stable_baselines3.common.callbacks import BaseCallback
import torch
import numpy as np
import os
import wandb

# Define your custom objective function for Optuna
def objective(trial):
    # Suggest hyperparameters for optimization
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    n_steps = trial.suggest_int('n_steps', 128, 2048)
    n_epochs = trial.suggest_int('n_epochs', 1, 20)
    gamma = trial.suggest_uniform('gamma', 0.9, 0.999)
    vf_coef = trial.suggest_uniform('vf_coef', 0.1, 0.9)
    clip_range = trial.suggest_uniform('clip_range', 0.1, 0.4)

    # Initialize the ClearML task
    task = Task.init(
        project_name='Mentor Group A/Group 1/Teun',
        task_name='Experiment with Optuna'
    )
    task.set_base_docker('deanis/2023y2b-rl:latest')
    task.execute_remotely(queue_name="default")

    # Initialize WandB project
    os.environ['WANDB_API_KEY'] = '17b671297e98466f9af4baa04230fcd84aec26c3'
    run = wandb.init(project="rl_experiment_9", sync_tensorboard=True)
    wandb.config.update({
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'n_steps': n_steps,
        'n_epochs': n_epochs,
        'gamma': gamma,
        'vf_coef': vf_coef,
        'clip_range': clip_range
    })

    try:
        training_env = OT2Env(render=False)
        check_env(training_env)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize the training environment: {e}")

    # PPO model initialization with trial parameters
    ppo_model = PPO(
        policy="MlpPolicy",
        env=training_env,
        verbose=1,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_steps=n_steps,
        n_epochs=n_epochs,
        gamma=gamma,
        ent_coef=vf_coef,
        clip_range=clip_range,
        tensorboard_log=f"tensorboard_logs/{run.id}",
    )

    # Define the callback for W&B logging and model saving
    wandb_callback_handler = WandbCallback(
        model_save_path=f"models/{run.id}",
        model_save_freq=10000,
        verbose=2
    )

    # Train the model
    total_training_steps = 200000
    ppo_model.learn(
        total_timesteps=total_training_steps,
        callback=[wandb_callback_handler]
    )

    # Evaluate the model performance
    rewards = []
    num_episodes = 10
    for _ in range(num_episodes):
        obs, _ = training_env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = ppo_model.predict(obs)
            obs, reward, done, _, _ = training_env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)

    mean_reward = np.mean(rewards)
    wandb.log({'mean_reward': mean_reward})

    return mean_reward

# Create and run the Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Save the best hyperparameters
best_hyperparams = study.best_params
print("Best hyperparameters: ", best_hyperparams)
