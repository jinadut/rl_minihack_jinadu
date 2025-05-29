import gymnasium as gym
from stable_baselines3 import PPO
import minihack_env as me
import os
from pathlib import Path

# Create models directory in home directory
home_dir = str(Path.home())
models_dir = os.path.join(home_dir, "rl_minihack_models")
os.makedirs(models_dir, exist_ok=True)

# Create environments
environments = {
    'EMPTY_ROOM': me.EMPTY_ROOM,
    'ROOM_WITH_MULTIPLE_MONSTERS': me.ROOM_WITH_MULTIPLE_MONSTERS
}

# Train and visualize for each environment
for env_name, env_id in environments.items():
    print(f"\nTraining on {env_name}")
    
    # Create environment
    env = me.get_minihack_envirnment(env_id, add_pixel=False, size=5)
    
    # Create and train model with MultiInputPolicy
    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4)
    
    # Save model in home directory
    model_path = os.path.join(models_dir, f"ppo_{env_name.lower()}")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Load model and visualize
    model = PPO.load(model_path)
    
    # Visualize the trained agent
    obs, info = env.reset()
    print(f"\nVisualizing trained agent in {env_name}:")
    print("=" * 50)
    
    for i in range(50):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Print the current state
        print(f"\nStep {i}:")
        print("-" * 30)
        print(f"Action taken: {action}")
        print(f"Reward: {reward}")
        print(f"Status: {'Terminated' if terminated else 'Truncated' if truncated else 'Continuing'}")
        
        if terminated or truncated:
            print("\nEpisode finished!")
            obs, info = env.reset()
            print("Starting new episode...")
            print("=" * 50)
    
    env.close()

print("\nTraining and visualization complete!")
