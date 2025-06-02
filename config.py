# config.py

# --- Shared parameters for all learning agents ---
AGENT_BASE_PARAMS = {
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.1,
    "epsilon_decay_strategy": "linear",
    "epsilon_decay_rate": 0.98,
    "q_table_default_value": 0.0
}

# --- Parameters specific to Q-Learning Agent ---
Q_LEARNING_AGENT_PARAMS = {
    **AGENT_BASE_PARAMS,
    "alpha": 0.3,
    "agent_id": "QLearningAgent"
}

# --- Parameters specific to SARSA (On-Policy TD) Agent ---
SARSA_AGENT_PARAMS = {
    **AGENT_BASE_PARAMS,
    "alpha": 0.3,
    "agent_id": "SARSAAgent"
}

# --- Parameters specific to Monte Carlo Agent ---
MONTE_CARLO_AGENT_PARAMS = {
    **AGENT_BASE_PARAMS,
    "first_visit": True,
    "agent_id": "MonteCarloAgent"
}

# --- Parameters specific to Dyna-Q Agent ---
DYNA_Q_AGENT_PARAMS = {
    **AGENT_BASE_PARAMS,
    "alpha": 0.3,
    "agent_id": "DynaQAgent",
    "planning_steps": 50
}

# --- Universal Reward Parameters ---
UNIVERSAL_REWARD_PARAMS = {
    "goal_reward": 1000,
    "step_penalty": -1.0,
    "death_penalty": -100
}

# --- General Training/Environment Parameters (can be expanded) ---
TRAINING_PARAMS = {
    "num_episodes_train": 3500,
    "num_episodes_visualize": 0,
    "max_steps_visualize": 100,

# --- Visualization Parameters --- #
    "visualize_in_training": False,
    "visualize_every_n_episodes": 100,
    "max_steps_for_viz_in_train": 1000,
    "max_episode_steps_global": 1000
}
