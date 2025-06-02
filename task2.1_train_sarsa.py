# task2.1_train_sarsa.py (derived from task2.1_train_mc.py)

import minihack_env as me
import agents
import commons
import config # Import your configuration
import matplotlib
# Attempt to use TkAgg backend for interactive plots, if available and tkinter is installed.
# This should be called before importing pyplot.
try:
    matplotlib.use('TkAgg')
except ImportError:
    print("TkAgg backend not available, using default backend. Plots might not be interactive or might save to file.")
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np # For np.mean if needed for interim logging


def main():
    print("Starting SARSA Agent training pipeline for multiple environments...")

    # --- Define Environments to Run ---
    # Ensure these constants (me.CLIFF, etc.) are correctly defined in your minihack_env.py
    environments_to_run = {
        "cliff": me.CLIFF,                 
        "empty_room": me.EMPTY_ROOM,       
        "room_with_lava": me.ROOM_WITH_LAVA,     
        "room_with_monster": me.ROOM_WITH_MONSTER
    }

    for env_name, env_id_val in environments_to_run.items():
        print(f"\n--- Starting Training for Environment: {env_name} ({env_id_val}) ---")

        # --- Configurable Parameters ---
        # These will be pulled from config.py, ensuring consistency
        max_steps_global = config.TRAINING_PARAMS.get("max_episode_steps_global", 1000) 
        num_train_episodes = config.TRAINING_PARAMS.get("num_episodes_train", 10000)
        viz_in_train = config.TRAINING_PARAMS.get("visualize_in_training", False)
        viz_every_n = config.TRAINING_PARAMS.get("visualize_every_n_episodes", 100)
        max_steps_viz_train = config.TRAINING_PARAMS.get("max_steps_for_viz_in_train", 1000)
        num_visualize_episodes_post = config.TRAINING_PARAMS.get("num_episodes_visualize", 1)
        max_steps_visualize_post = config.TRAINING_PARAMS.get("max_steps_visualize", 1000)

        # --- Environment Setup ---
        obs_keys = ['chars', 'pixel'] 
        env = me.get_minihack_envirnment(
            env_id_val,
            observation_keys=obs_keys,
            add_pixel=True, 
            max_episode_steps=max_steps_global
        )

        # --- Agent Setup ---
        sarsa_agent = agents.SARSAAgent(
            action_space=env.action_space, 
            params=config.SARSA_AGENT_PARAMS
        )

        # --- Task Setup ---
        task = commons.AbstractRLTask(env, sarsa_agent)

        # --- Training ---
        print(f"Training {sarsa_agent.id} on {env_name} for {num_train_episodes} episodes.")
        if viz_in_train:
            print(f"In-training visualization: Every {viz_every_n} episodes, for up to {max_steps_viz_train} steps.")

        avg_returns_train = task.interact(
            n_episodes=num_train_episodes,
            visualize_training_episodes=viz_in_train,
            visualize_every_n_episodes=viz_every_n,
            max_steps_for_viz_in_train=max_steps_viz_train
        )
        
        print(f"Training on {env_name} finished. Final epsilon for {sarsa_agent.id}: {sarsa_agent.epsilon:.4f}")

        # --- Plotting Average Returns ---
        plt.figure(figsize=(12, 6))
        plt.plot(avg_returns_train, label=f'{sarsa_agent.id} Average Returns on {env_name}')
        plt.title(f'SARSA Agent Training on {env_name}')
        plt.xlabel('Episode')
        plt.ylabel('Average Return (Smoothed)')
        plt.legend()
        plt.grid(True)
        
        # Define the path to your macOS home as it's mounted in the VM
        mac_home_in_vm = "/Users/thiesjinadu"  # Path to macOS home inside the VM

        # Construct the output directory path based on the mounted macOS home
        agent_name_for_path = sarsa_agent.id.lower().replace("agent", "").replace("-","") # e.g., "sarsa"
        if not agent_name_for_path: agent_name_for_path = "sarsa_default" # Fallback
        
        task_subfolder = f"2.1_{agent_name_for_path}" # e.g., "2.1_sarsa"

        output_dir_base = os.path.join(mac_home_in_vm, "Documents", "rl_results")
        output_dir_task_specific = os.path.join(output_dir_base, task_subfolder, env_name)

        os.makedirs(output_dir_task_specific, exist_ok=True)
        
        plot_filename = os.path.join(output_dir_task_specific, f"average_returns_{agent_name_for_path}_{env_name}.png")
        plt.savefig(plot_filename)
        print(f"Average returns plot for {env_name} saved to {plot_filename} (on macOS host)")
        
        try:
            plt.show(block=False)
            plt.pause(1) 
            plt.close() 
        except Exception as e:
            print(f"Could not show average returns plot for {env_name} interactively: {e}")
            plt.close()

        # --- Visualization of Trained Agent (Post-Training) ---
        if num_visualize_episodes_post > 0:
            print(f"\nVisualizing (post-training) {sarsa_agent.id} on {env_name} for {num_visualize_episodes_post} episode(s), up to {max_steps_visualize_post} steps each...")
            for i in range(num_visualize_episodes_post):
                print(f"--- Post-Training Visualization Episode {i+1}/{num_visualize_episodes_post} on {env_name} ---")
                task.visualize_episode(max_number_steps=max_steps_visualize_post)

        # --- Cleanup for the current environment ---
        plt.close('all') 
        env.close()
        print(f"\n--- Completed Training and Visualization for Environment: {env_name} ---")

    print("\nSARSA agent training pipeline complete for all specified environments.")

if __name__ == "__main__":
    main() 