import gymnasium as gym
from stable_baselines3 import DQN
import minihack_env as me
import os
from pathlib import Path
from nle import nethack
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

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

class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_position = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_position = self._get_position(obs)
        return obs, info
        
    def _get_position(self, obs):
        # Extract position from observation
        if isinstance(obs, dict) and 'chars' in obs:
            chars = obs['chars']
            # Find the '@' symbol which represents the agent
            agent_pos = np.where(chars == ord('@'))
            if len(agent_pos[0]) > 0:
                return (agent_pos[0][0], agent_pos[1][0])
        return None
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add reward shaping
        current_position = self._get_position(obs)
        if self.last_position is not None and current_position is not None:
            # Reward for moving (small positive reward)
            if current_position != self.last_position:
                reward += 0.1
            # Penalty for staying in the same place (to prevent getting stuck)
            else:
                reward -= 0.2
                
        self.last_position = current_position
        return obs, reward, terminated, truncated, info

# Create environments
environments = {
    'EMPTY_ROOM': me.EMPTY_ROOM,
    'ROOM_WITH_MULTIPLE_MONSTERS': me.ROOM_WITH_MULTIPLE_MONSTERS
}

# Train and visualize for each environment
for env_name, env_id in environments.items():
    print(f"\nTraining on {env_name}")
    
    # Create environment with reward shaping
    env = me.get_minihack_envirnment(env_id, add_pixel=False, size=5)
    env = RewardShapingWrapper(env)
       
    # Create and train model with MultiInputPolicy
    model = DQN(
        "MultiInputPolicy", 
        env, 
        verbose=1,
        learning_rate=5e-4,  # Increased learning rate
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.3,  # Increased exploration time
        exploration_initial_eps=1.0,
        exploration_final_eps=0.2,  # Increased final exploration
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        policy_kwargs=dict(
            net_arch=[128, 128]  # Added a hidden layer
        )
    )
    
    # Train with callback
    model.learn(
        total_timesteps=50000, 
        log_interval=4,
    )
    
    # Save model in home directory
    model_path = os.path.join(models_dir, f"dqn_{env_name.lower()}")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Load model and visualize
    model = DQN.load(model_path)
    
    # Visualize the trained agent
    obs, info = env.reset()
    print(f"\nVisualizing trained agent in {env_name}:")
    print("=" * 50)
    
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
            obs, info = env.reset()
            print("Starting new episode...")
            print("=" * 50)
    
    # Print final action distribution
    total = sum(action_counts.values())
    print("\nFinal Action Distribution:")
    for action, count in action_counts.items():
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"{ACTION_NAMES[action]}: {percentage:.1f}%")
    
    env.close()

print("\nTraining and visualization complete!")