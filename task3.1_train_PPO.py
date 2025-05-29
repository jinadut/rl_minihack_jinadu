import gymnasium as gym
import minihack
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor # To ensure SB3 can log episode stats

from commons import PixelObservationWrapper
from config import PPO_SB3_PARAMS, SB3_TRAINING_PARAMS, SB3_ENV_IDS

import os

def train_ppo_agent(env_id: str, model_save_path: str, tensorboard_log_path: str):
    """
    Trains a PPO agent on the specified MiniHack environment.

    :param env_id: The ID of the MiniHack environment.
    :param model_save_path: Path to save the trained model.
    :param tensorboard_log_path: Path to save TensorBoard logs.
    """
    print(f"--- Training PPO on {env_id} ---")

    # 1. Create and wrap the environment
    env = gym.make(env_id)
    env = PixelObservationWrapper(env)
    env = Monitor(env) # Wrap with Monitor for SB3 logging
    env = DummyVecEnv([lambda: env]) # SB3 requires a VecEnv

    # 2. Instantiate the PPO agent
    # Ensure tensorboard_log path exists
    os.makedirs(tensorboard_log_path, exist_ok=True)

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=tensorboard_log_path,
        **PPO_SB3_PARAMS
    )

    # 3. Train the agent
    print(f"Starting training for {SB3_TRAINING_PARAMS['total_timesteps_train']} timesteps...")
    model.learn(
        total_timesteps=SB3_TRAINING_PARAMS['total_timesteps_train'],
        log_interval=SB3_TRAINING_PARAMS.get('log_interval', 1) # PPO's log_interval is number of updates
    )
    print("Training complete.")

    # 4. Save the model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # 5. Evaluate the trained model
    print("Evaluating trained model...")
    eval_env = gym.make(env_id)
    eval_env = PixelObservationWrapper(eval_env)
    eval_env = Monitor(eval_env)
    
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=SB3_TRAINING_PARAMS['n_eval_episodes'],
        deterministic=True
    )
    print(f"Evaluation results for {env_id}:")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # 6. Close the environment
    env.close()
    eval_env.close()
    print(f"--- Finished PPO training and evaluation for {env_id} ---\n")
    return mean_reward, std_reward

if __name__ == "__main__":
    # Create directories for models and logs if they don't exist
    os.makedirs("models/ppo", exist_ok=True)
    os.makedirs("logs/ppo_tensorboard", exist_ok=True)

    # Define the environments to train for this specific (PPO) task
    environments_to_train_ppo = {
        "empty_room": SB3_ENV_IDS["empty_room"],
        "multiple_monsters_quest": SB3_ENV_IDS["multiple_monsters_quest"]
    }

    results = {}

    for env_name, env_id_str in environments_to_train_ppo.items():
        print(f"Starting PPO process for environment: {env_name} ({env_id_str})")
        model_path = f"models/ppo/ppo_{env_name.replace('-', '_')}_model.zip"
        tb_log_path = f"logs/ppo_tensorboard/ppo_{env_name.replace('-', '_')}_logs/"
        
        mean_r, std_r = train_ppo_agent(env_id_str, model_path, tb_log_path)
        results[env_name] = {"mean_reward": mean_r, "std_reward": std_r}

    print("\n--- Overall PPO Training Summary ---")
    for env_name, res in results.items():
        print(f"Environment: {env_name} - Mean Reward: {res['mean_reward']:.2f} +/- {res['std_reward']:.2f}")
