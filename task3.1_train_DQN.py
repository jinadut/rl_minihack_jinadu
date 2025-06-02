import gymnasium as gym
from stable_baselines3 import DQN
import minihack_env as me
import os
from pathlib import Path
from nle import nethack
import numpy as np

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
    'EMPTY_ROOM': me.EMPTY_ROOM
    #'ROOM_WITH_MULTIPLE_MONSTERS': me.ROOM_WITH_MULTIPLE_MONSTERS
}

# Train and visualize for each environment
for env_name, env_id in environments.items():
    print(f"\nTraining on {env_name}")
    
    # Create environment with reward shaping
    env = me.get_minihack_envirnment(env_id, add_pixel=False)
       
    # Create and train model with MultiInputPolicy
    model = DQN(
        "MultiInputPolicy", 
        env, 
        verbose=1,
        learning_rate=0.00015,
        buffer_size=50000,
        learning_starts=5000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.25,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        policy_kwargs=dict( # Added policy_kwargs for a more complex network
            net_arch=[128, 128] # Two hidden layers with 128 neurons each
        )
    )
    
    # Train with callback
    # Instantiate the custom callback
    # positive_reward_logger = PositiveRewardLoggerCallback()
    
    model.learn(
        total_timesteps=100000, 
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
    print(f"\nVisualizing trained agent in {env_name}:")
    print("INITIAL OBSERVATION CHARS GRID:")
    if isinstance(obs, dict) and 'chars' in obs:
        for r_idx, r_val in enumerate(obs['chars']):
            try:
                print(f"Row {r_idx:2d}: {''.join([chr(c) if c < 256 else '?' for c in r_val])}")
            except TypeError:
                print(f"Row {r_idx:2d}: Error converting row to chars - {r_val}")
    else:
        print("obs['chars'] not found or obs is not a dict.")
    print("=" * 50)
    
    action_counts = {i: 0 for i in range(4)}
    
    for i in range(50):
        # Print the current observation (or parts of it) before prediction
        print(f"\n--- Obs for Step {i} ---")
        if isinstance(obs, dict):
            if 'chars' in obs:
                # Try to find agent position to print a local view
                agent_pos = np.where(obs['chars'] == ord('@'))
                if len(agent_pos[0]) > 0:
                    r, c = agent_pos[0][0], agent_pos[1][0]
                    # Print a 5x5 window around the agent, converting bytes to chars
                    local_view = obs['chars'][max(0, r-2):r+3, max(0, c-2):c+3]
                    print("Local view (chars around @):")
                    for row_bytes in local_view:
                        print(" ".join([chr(b) for b in row_bytes])) # Decode bytes to characters
                else:
                    print("Agent '@' not found in obs['chars']")
        else:
            print(f"Raw obs: {obs}") # Fallback for non-dict observations
        print("----------------------")

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
        print(f"Observation: {obs}")
        
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