import gymnasium as gym
from stable_baselines3 import DQN
import minihack_env as me
import os
from pathlib import Path
from nle import nethack
import numpy as np
from gymnasium.wrappers import FlattenObservation

# Define the base path for saving models as per user request
# (model name) will be "DQN", (environment name) will be from the loop
USER_SAVE_BASE_PATH = "/Users/thiesjinadu/Documents/"
MODEL_TYPE_FOLDER_NAME = "DQN" # This represents the (model name) part of the path

# Create models directory in home directory
home_dir = str(Path.home())
models_dir = os.path.join(home_dir, "rl_minihack_models")
os.makedirs(models_dir, exist_ok=True)

# Define action mapping
ACTION_NAMES = {
    0: "North",
    1: "East",
    2: "South",
    3: "West"
}

# Create environments
environments = {
    'EMPTY_ROOM': me.EMPTY_ROOM,
    'ROOM_WITH_MULTIPLE_MONSTERS': me.ROOM_WITH_MULTIPLE_MONSTERS
}

# Train and visualize for each environment
for env_name, env_id in environments.items():
    print(f"\nTraining on {env_name}")
    
    # Create environment with reward shaping
    env = me.get_minihack_envirnment(env_id, add_pixel=False)
    env = FlattenObservation(env)
       
    # Create and train model with MultiInputPolicy
    model = DQN(
        "MlpPolicy",
        env, 
        verbose=1,
        learning_rate=0.0001,
        buffer_size=1500,
        learning_starts=5000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.25,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        train_freq=4,
        #gradient_steps=1,
        target_update_interval=1000,
        policy_kwargs=dict( # Added policy_kwargs for a more complex network
            net_arch=[128, 128] # Two hidden layers with 128 neurons each
        )
    )
    # Train with callback
    # Instantiate the custom callback
    # positive_reward_logger = PositiveRewardLoggerCallback()
    
    model.learn(
        total_timesteps=1000000,
        log_interval=4
    )
    
    # Create specific directory for this model and environment
    # env_name will be like 'EMPTY_ROOM' or 'ROOM_WITH_MULTIPLE_MONSTERS'
    current_model_env_dir = os.path.join(USER_SAVE_BASE_PATH, MODEL_TYPE_FOLDER_NAME, env_name)
    os.makedirs(current_model_env_dir, exist_ok=True)

    # Define model save path and save model
    # The filename will be like 'dqn_empty_room'
    # Stable Baselines3 will append '.zip' to this path when saving.
    model_filename = f"dqn_{env_name.lower()}"
    model_path = os.path.join(current_model_env_dir, model_filename)
    model.save(model_path)
    print(f"Model saved to {model_path}.zip") # Clarify .zip in print
    
    # Load model and visualize
    model = DQN.load(model_path) # SB3 handles .zip for loading too
    
    # Visualize the trained agent
    obs, info = env.reset()
    action_counts = {i: 0 for i in range(4)}
    
    for i in range(50):

        action, _states = model.predict(obs, deterministic=True)
        # Convert numpy array to integer - handle both 0D and 1D arrays
        action = int(action.item()) if isinstance(action, np.ndarray) else int(action)
        action_counts[action] += 1
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Print the current state
        print(f"\nStep {i}:")
        print("-" * 30)

        print(f"Action taken: {action} ({ACTION_NAMES.get(action, 'Unknown')})")
        print(f"Reward: {reward}")
        print(f"Status: {'Terminated' if terminated else 'Truncated' if truncated else 'Continuing'}")

        
        if terminated or truncated:
            print("\nEpisode finished!")
            print(f"Final Info from env.step(): {info}")
            break
    
    # Print final action distribution
    total = sum(action_counts.values())
    print("\nFinal Action Distribution:")
    for action, count in action_counts.items():
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"{ACTION_NAMES[action]}: {percentage:.1f}%")
    
    env.close()

print("\nTraining and visualization complete!")