import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

from stable_baselines3 import A2C

# Create environment with rgb_array render mode
env = gym.make("CartPole-v1", render_mode="rgb_array")

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()

# Create a figure for visualization
plt.figure(figsize=(8, 6))

for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    
    # Get the rendered frame
    frame = vec_env.render()
    
    # Display the frame using matplotlib
    plt.clf()
    plt.imshow(frame)
    plt.axis('off')
    plt.title(f'Step {i}')
    plt.pause(0.01)  # Small pause to allow the plot to update
    
    if done:
        obs = vec_env.reset()

plt.close()