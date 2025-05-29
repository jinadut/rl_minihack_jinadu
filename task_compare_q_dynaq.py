# task_compare_q_dynaq.py

import minihack_env as me
import agents
import commons
import config
import matplotlib
try:
    matplotlib.use('TkAgg')
except ImportError:
    print("TkAgg backend not available, using default backend. Plots might not be interactive or might save to file.")
import matplotlib.pyplot as plt
import os
import numpy as np

def main():
    print("Starting Q-learning vs Dyna-Q comparison...")

    environments_to_run = {
        "room_with_monster": me.ROOM_WITH_MONSTER,
        "cliff": me.CLIFF
    }

    agents_to_compare = {
        "QLearningAgent": (agents.QLearningAgent, config.Q_LEARNING_AGENT_PARAMS),
        "DynaQAgent": (agents.DynaQAgent, config.DYNA_Q_AGENT_PARAMS)
    }

    # Pull general training parameters from config
    max_steps_global = config.TRAINING_PARAMS.get("max_episode_steps_global", 500)
    num_train_episodes = config.TRAINING_PARAMS.get("num_episodes_train", 5000)
    viz_in_train = config.TRAINING_PARAMS.get("visualize_in_training", False)
    viz_every_n = config.TRAINING_PARAMS.get("visualize_every_n_episodes", 100)
    max_steps_viz_train = config.TRAINING_PARAMS.get("max_steps_for_viz_in_train", 1000)
    num_visualize_episodes_post = config.TRAINING_PARAMS.get("num_episodes_visualize", 0) # Disable post-viz for comparison script by default
    max_steps_visualize_post = config.TRAINING_PARAMS.get("max_steps_visualize", 1000)

    for env_name, env_id_val in environments_to_run.items():
        print(f"\n--- Comparing Agents on Environment: {env_name} ({env_id_val}) ---")
        
        plt.figure(figsize=(14, 7))
        plt.title(f'Agent Comparison on {env_name}')
        plt.xlabel('Episode')
        plt.ylabel('Average Return (Smoothed)')

        all_avg_returns_for_env = {}

        for agent_name, (agent_class, agent_params) in agents_to_compare.items():
            print(f"  -- Running {agent_name} --")
            
            # --- Environment Setup (re-initialize for each agent to ensure fair comparison) ---
            obs_keys = ['chars', 'blstats', 'pixel'] 
            env = me.get_minihack_envirnment(
                env_id_val,
                observation_keys=obs_keys,
                add_pixel=True, 
                max_episode_steps=max_steps_global
            )

            # --- Agent Setup ---
            agent_instance = agent_class(
                action_space=env.action_space, 
                params=agent_params
            )

            # --- Task Setup ---
            task = commons.AbstractRLTask(env, agent_instance)

            # --- Training ---
            print(f"    Training {agent_instance.id} on {env_name} for {num_train_episodes} episodes.")
            if viz_in_train: # Note: viz_in_train will run for each agent if enabled
                print(f"    In-training visualization: Every {viz_every_n} episodes, for up to {max_steps_viz_train} steps.")

            avg_returns_train = task.interact(
                n_episodes=num_train_episodes,
                visualize_training_episodes=viz_in_train,
                visualize_every_n_episodes=viz_every_n,
                max_steps_for_viz_in_train=max_steps_viz_train
            )
            all_avg_returns_for_env[agent_name] = avg_returns_train
            
            print(f"    Training for {agent_instance.id} on {env_name} finished. Final epsilon: {agent_instance.epsilon:.4f}")
            
            # Plot this agent's results on the combined figure for the current environment
            plt.plot(avg_returns_train, label=f'{agent_instance.id}')

            if num_visualize_episodes_post > 0:
                print(f"\n    Visualizing (post-training) {agent_instance.id} on {env_name} for {num_visualize_episodes_post} episode(s)...")
                for i in range(num_visualize_episodes_post):
                    task.visualize_episode(max_number_steps=max_steps_visualize_post)
            
            env.close() # Close env for this agent before starting next

        # Finalize and save the plot for the current environment
        plt.legend()
        plt.grid(True)
        
        output_dir = f"results/compare_q_dynaq/"
        os.makedirs(output_dir, exist_ok=True)
        plot_filename = os.path.join(output_dir, f"comparison_{env_name}.png")
        plt.savefig(plot_filename)
        print(f"  Comparison plot for {env_name} saved to {plot_filename}")
        
        try:
            plt.show(block=False)
            plt.pause(1) 
            plt.close()
        except Exception as e:
            print(f"  Could not show comparison plot for {env_name} interactively: {e}")
            plt.close()

        plt.close('all') # Close any other figures that might be open

    print("\nQ-learning vs Dyna-Q comparison complete.")

if __name__ == "__main__":
    main() 