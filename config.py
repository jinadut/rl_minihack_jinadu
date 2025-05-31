# config.py

# --- Shared parameters for all learning agents ---
AGENT_BASE_PARAMS = {
    "gamma": 0.995,              # Higher discount factor for better reward propagation
    "epsilon_start": 1.0,       # Initial exploration rate (also used as start for linear decay)
    "epsilon_min": 0.05,        # Minimum exploration rate
    "epsilon_decay_strategy": "linear", # Options: "linear", "multiplicative"
    "epsilon_decay_rate": 0.98, # Multiplicative decay factor (only used if strategy is "multiplicative")
    "q_table_default_value": 0.0 # Changed from 100.0 to 0.0 - neutral initialization
}

# --- Parameters specific to Q-Learning Agent ---
Q_LEARNING_AGENT_PARAMS = {
    **AGENT_BASE_PARAMS, # Inherit base parameters
    "alpha": 0.2,              # Decreased from 0.3 for more stable learning
    "agent_id": "QLearningAgent"
}

# --- Parameters specific to SARSA (On-Policy TD) Agent ---
SARSA_AGENT_PARAMS = {
    **AGENT_BASE_PARAMS, # Inherit base parameters
    "alpha": 0.1,              # Increased from 0.15 to 0.3 for faster learning
    "agent_id": "SARSAAgent"
}

# --- Parameters specific to Monte Carlo Agent ---
MONTE_CARLO_AGENT_PARAMS = {
    **AGENT_BASE_PARAMS, # Inherit base parameters
    "first_visit": True,       # True for First-Visit MC, False for Every-Visit MC
    "agent_id": "MonteCarloAgent"
}

# --- Parameters specific to Dyna-Q Agent ---
DYNA_Q_AGENT_PARAMS = {
    **AGENT_BASE_PARAMS,  # Inherit base parameters
    "alpha": 0.2,           # Increased from 0.15 to 0.3 for faster learning
    "agent_id": "DynaQAgent",
    "planning_steps": 50    # Number of simulated experiences per real experience
}

# --- Universal Reward Parameters ---
UNIVERSAL_REWARD_PARAMS = {
    "goal_reward": 5000,       # Reward for achieving the task/goal
    "step_penalty": -1.0,        # Penalty for each step taken
    "death_penalty": -100      # Adjusted death penalty
}

# --- General Training/Environment Parameters (can be expanded) ---
TRAINING_PARAMS = {
    "num_episodes_train": 500,   # Increased from 450 episodes for training runs
    "num_episodes_visualize": 0,      # Number o§f episodes for post-training visualization
    "max_steps_visualize": 1000,       # Max steps per episode during post-training visualization
    
    # New parameters for in-training visualization
    "visualize_in_training": False,      # Master switch for in-training visualization
    "visualize_every_n_episodes": 100,   # Visualize episode 1, then every Nth episode (e.g., 1, 101, 201...)
    "max_steps_for_viz_in_train": 1000,    # Max steps to show for an episode visualized during training

    "max_episode_steps_global": 4000    # Global max steps for any environment episode if not otherwise specified by env itself
}
