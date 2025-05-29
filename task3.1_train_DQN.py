import gymnasium as gym
import minihack
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor # To ensure SB3 can log episode stats

from commons import PixelObservationWrapper
from config import DQN_SB3_PARAMS, SB3_TRAINING_PARAMS, SB3_ENV_IDS

import os

def train_dqn_agent(env_id: str, model_save_path: str, tensorboard_log_path: str):
    """
    Trains a DQN agent on the specified MiniHack environment.

    :param env_id: The ID of the MiniHack environment.
    :param model_save_path: Path to save the trained model.
    :param tensorboard_log_path: Path to save TensorBoard logs.
    """
    print(f"--- Training DQN on {env_id} ---")

    # 1. Create and wrap the environment
    env = gym.make(env_id)
    env = PixelObservationWrapper(env)
    env = Monitor(env) # Wrap with Monitor for SB3 logging
    env = DummyVecEnv([lambda: env]) # SB3 requires a VecEnv

    # 2. Instantiate the DQN agent
    # Ensure tensorboard_log path exists
    os.makedirs(tensorboard_log_path, exist_ok=True)
    
    model = DQN(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=tensorboard_log_path,
        **DQN_SB3_PARAMS
    )

    # 3. Train the agent
    print(f"Starting training for {SB3_TRAINING_PARAMS['total_timesteps_train']} timesteps...")
    model.learn(
        total_timesteps=SB3_TRAINING_PARAMS['total_timesteps_train'],
        log_interval=4 # DQN's learn log_interval is for episodes
    )
    print("Training complete.")

    # 4. Save the model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # 5. Evaluate the trained model
    print("Evaluating trained model...")
    # For evaluation, create a fresh, possibly non-vectorized env (or a new VecEnv)
    eval_env = gym.make(env_id)
    eval_env = PixelObservationWrapper(eval_env)
    eval_env = Monitor(eval_env) # Monitor for evaluation too
    # No DummyVecEnv needed for evaluate_policy if using a single env instance
    
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=SB3_TRAINING_PARAMS['n_eval_episodes'],
        deterministic=True # Usually evaluate with deterministic actions
    )
    print(f"Evaluation results for {env_id}:")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # 6. Close the environment
    env.close()
    eval_env.close()
    print(f"--- Finished DQN training and evaluation for {env_id} ---\n")
    return mean_reward, std_reward

if __name__ == "__main__":
    # Create directories for models and logs if they don't exist
    os.makedirs("models/dqn", exist_ok=True)
    os.makedirs("logs/dqn_tensorboard", exist_ok=True)

    # Define the environments to train for this specific (DQN) task
    environments_to_train_dqn = {
        "empty_room": SB3_ENV_IDS["empty_room"],
        "multiple_monsters_quest": SB3_ENV_IDS["multiple_monsters_quest"]
    }

    results = {}

    for env_name, env_id_str in environments_to_train_dqn.items():
        print(f"Starting DQN process for environment: {env_name} ({env_id_str})")
        model_path = f"models/dqn/dqn_{env_name.replace('-', '_')}_model.zip"
        tb_log_path = f"logs/dqn_tensorboard/dqn_{env_name.replace('-', '_')}_logs/"
        
        mean_r, std_r = train_dqn_agent(env_id_str, model_path, tb_log_path)
        results[env_name] = {"mean_reward": mean_r, "std_reward": std_r}

    print("\n--- Overall DQN Training Summary ---")
    for env_name, res in results.items():
        print(f"Environment: {env_name} - Mean Reward: {res['mean_reward']:.2f} +/- {res['std_reward']:.2f}")
