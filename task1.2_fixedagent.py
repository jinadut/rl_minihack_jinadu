import minihack_env as me
import agents
import commons
import matplotlib.pyplot as plt # Though not strictly needed if visualize_episode only prints
import os
import config # Import config

results_dir_base = "results/1.2/"
os.makedirs(results_dir_base, exist_ok=True)


def run_fixed_agent_on_env(env_id_str, agent_instance, max_steps_viz, max_episode_steps_env):
    """Helper function to run FixedAgent on a given environment and visualize.
    The environment name is derived from the env_id_str for saving purposes.
    """
    print(f"\n--- Running FixedAgent on {env_id_str} (max_env_steps: {max_episode_steps_env}) ---")
    # env_name = env_id_str.split('-')[-1] # e.g., "emptyroom", "roomwithlava"
    
    obs_keys = ['chars', 'blstats', 'pixel'] 
    env = me.get_minihack_envirnment(
        env_id_str, 
        observation_keys=obs_keys, 
        add_pixel=True,
        max_episode_steps=max_episode_steps_env # Pass configured max steps
    )

    task = commons.AbstractRLTask(env, agent_instance)
    agent_instance.onEpisodeEnd()

    print(f"Visualizing {max_steps_viz} timesteps...")
    task.visualize_episode(max_number_steps=max_steps_viz)
    
    env.close()
    print(f"--- Finished FixedAgent on {env_id_str} ---")

if __name__ == "__main__":
    environments_to_test = [
        me.EMPTY_ROOM, 
        me.ROOM_WITH_LAVA
    ]
    
    # Get global max episode steps from config
    max_ep_steps_global_cfg = config.TRAINING_PARAMS.get("max_episode_steps_global", 200)
    # Max steps for visualization of fixed agent (can be different from in-training viz)
    max_steps_fixed_agent_viz = 10 # Or get from config if you add it there

    try:
        obs_keys_for_dummy = ['chars', 'blstats', 'pixel'] 
        dummy_env_for_action_space = me.get_minihack_envirnment(
            me.EMPTY_ROOM, 
            observation_keys=obs_keys_for_dummy, 
            add_pixel=True,
            max_episode_steps=max_ep_steps_global_cfg # Use configured max steps for dummy env too
        )
        action_space = dummy_env_for_action_space.action_space
        dummy_env_for_action_space.close()
    except Exception as e:
        print(f"Could not create dummy env for action space, error: {e}")
        print("Exiting. Ensure minihack_env.py can create EMPTY_ROOM environment.")
        exit()

    fixed_agent = agents.FixedAgent(action_space)

    for env_id in environments_to_test:
        run_fixed_agent_on_env(env_id, fixed_agent, 
                               max_steps_viz=max_steps_fixed_agent_viz, 
                               max_episode_steps_env=max_ep_steps_global_cfg)

    print("\nAll FixedAgent visualizations complete.") 