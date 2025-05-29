# config.py

# --- Shared parameters for all learning agents ---
AGENT_BASE_PARAMS = {
    "gamma": 0.98,              # Discount factor
    "epsilon_start": 1.0,       # Initial exploration rate (also used as start for linear decay)
    "epsilon_min": 0.01,        # Minimum exploration rate
    "epsilon_decay_strategy": "linear", # Options: "linear", "multiplicative"
    "epsilon_decay_rate": 0.995, # Multiplicative decay factor (only used if strategy is "multiplicative")
    "q_table_default_value": 0.0 # Default value for new Q-table entries
}

# --- Parameters specific to Q-Learning Agent ---
Q_LEARNING_AGENT_PARAMS = {
    **AGENT_BASE_PARAMS, # Inherit base parameters
    "alpha": 0.1,              # Learning rate
    "agent_id": "QLearningAgent"
}

# --- Parameters specific to SARSA (On-Policy TD) Agent ---
SARSA_AGENT_PARAMS = {
    **AGENT_BASE_PARAMS, # Inherit base parameters
    "alpha": 0.05,              # Learning rate
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
    "alpha": 0.1,           # Learning rate for Q-learning updates (real and simulated)
    "agent_id": "DynaQAgent",
    "planning_steps": 50    # Number of simulated experiences per real experience
}

# --- Universal Reward Parameters ---
UNIVERSAL_REWARD_PARAMS = {
    "goal_reward": 10000000.0,       # Reward for achieving the task/goal
    "step_penalty": -1.0,        # Penalty for each step taken
    "death_penalty": -2000       # Penalty for dying / failing task
}

# --- General Training/Environment Parameters (can be expanded) ---
TRAINING_PARAMS = {
    "num_episodes_train": 1000,   # Number of episodes for training runs
    "num_episodes_visualize": 1,      # Number of episodes for post-training visualization
    "max_steps_visualize": 1000,       # Max steps per episode during post-training visualization
    
    # New parameters for in-training visualization
    "visualize_in_training": False,      # Master switch for in-training visualization
    "visualize_every_n_episodes": 100,   # Visualize episode 1, then every Nth episode (e.g., 1, 101, 201...)
    "max_steps_for_viz_in_train": 1000,    # Max steps to show for an episode visualized during training

    "max_episode_steps_global": 700    # NEW: Global max steps for any environment episode if not otherwise specified by env itself
}

# Example of how to access:
# from config import Q_LEARNING_AGENT_PARAMS
# alpha = Q_LEARNING_AGENT_PARAMS["alpha"]

# --- Parameters for Stable Baselines 3 Agents ---

DQN_SB3_PARAMS = {
    "learning_rate": 1e-4,
    "buffer_size": 50000,       # populaires: 50k, 100k
    "learning_starts": 1000,    # Timesteps before learning starts
    "batch_size": 32,
    "tau": 1.0,                 # Polyak update coefficient
    "gamma": 0.99,              # Discount factor
    "train_freq": 4,            # Update the model every 4 steps
    "gradient_steps": 1,        # How many gradient steps to perform after each update
    # For CnnPolicy, policy_kwargs can specify CNN architecture if needed
    # "policy_kwargs": dict(net_arch=[64, 64]) # Example for MlpPolicy, CnnPolicy has its own arch.
}

PPO_SB3_PARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,           # Number of steps to run for each environment per update
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,           # Entropy coefficient for exploration
    "vf_coef": 0.5,             # Value function coefficient
    # For CnnPolicy, policy_kwargs can specify CNN architecture if needed
}

SB3_TRAINING_PARAMS = {
    "total_timesteps_train": 100000, # Total timesteps for training (e.g., 100k, 200k)
    "n_eval_episodes": 20,           # Number of episodes for evaluation
    "log_interval": 10,               # Log training info every N episodes (for PPO/A2C during .learn())
                                     # For DQN, .learn() has its own log_interval for Tensorboard
}

# Define environment IDs to be used by SB3 scripts
SB3_ENV_IDS = {
    "empty_room": "MiniHack-Room-5x5-v0",
    "multiple_monsters_quest": "MiniHack-Quest-Medium-v0"
    # Add more if needed, e.g. "MiniHack-CorridorBattle-v0"
}
