#!/usr/bin/env python3
"""
Test all four RL agents including DynaQ.
"""

import minihack_env as me
import numpy as np
from agents import QLearningAgent, SARSAAgent, MonteCarloAgent, DynaQAgent
# Import specific param sets if we want to modify them for a specific agent
# Also import UNIVERSAL_REWARD_PARAMS for consistent reward settings with training.
from config import Q_LEARNING_AGENT_PARAMS, SARSA_AGENT_PARAMS, MONTE_CARLO_AGENT_PARAMS, DYNA_Q_AGENT_PARAMS, UNIVERSAL_REWARD_PARAMS
import commons # For AbstractRLTask

def test_all_agents():
    """Test all four agents, ensuring test setup aligns with AbstractRLTask usage in training."""
    print("=== TESTING ALL AGENTS on Room with Lava (Aligned with AbstractRLTask) ===")
    
    q_learning_test_params = {
        **Q_LEARNING_AGENT_PARAMS,
        "alpha": 0.2,
        "epsilon_min": 0.01,
        "epsilon_decay_strategy": "linear"
    }

    agents_to_test = [
        (QLearningAgent, "Q-LEARNING", q_learning_test_params),
        (SARSAAgent, "SARSA", SARSA_AGENT_PARAMS),
        (MonteCarloAgent, "MONTE CARLO", MONTE_CARLO_AGENT_PARAMS),
        (DynaQAgent, "DYNA-Q", DYNA_Q_AGENT_PARAMS)
    ]

    results = {}
    
    test_env_id =  me.EMPTY_ROOM #me.ROOM_WITH_LAVA me.EMPTY_ROOM
    test_env_id_str = "empty_room" #"room-with-lava"
    test_max_episode_steps = 150
    test_num_episodes = 1000

    print(f"--- Test Parameters ---")
    print(f"Environment: {test_env_id_str}")
    print(f"Num Episodes: {test_num_episodes}")
    print(f"Max Steps per Episode: {test_max_episode_steps}")
    
    goal_reward = UNIVERSAL_REWARD_PARAMS.get("goal_reward", 1.0)
    death_penalty = UNIVERSAL_REWARD_PARAMS.get("death_penalty", -1.0)
    step_penalty = UNIVERSAL_REWARD_PARAMS.get("step_penalty", -0.01)

    print(f"Goal Reward: {goal_reward}")
    print(f"Death Penalty: {death_penalty}")
    print(f"Step Penalty: {step_penalty}")
    
    # Define success threshold: e.g., total reward > 0 
    # (assuming goal_reward is positive and significant compared to accumulated step penalties for a successful run)
    success_reward_threshold = 0.0 
    print(f"Success Reward Threshold (total episode reward > {success_reward_threshold})")
    print(f"--- Starting Tests ---")

    for agent_class, name, agent_params_config in agents_to_test:
        print(f"\n--- Testing {name} on {test_env_id_str} for {test_num_episodes} episodes (max {test_max_episode_steps} steps/ep) ---")
        
        # Create a temporary environment to get action space if needed by agent constructor
        # This is done to avoid instantiating the main env multiple times if action space is complex to get.
        # However, it's often cleaner to get action_space from the main env instance.
        temp_env_for_action_space = me.get_minihack_envirnment(test_env_id, observation_keys=['chars'])
        agent = agent_class(action_space=temp_env_for_action_space.action_space, params=agent_params_config)
        temp_env_for_action_space.close()

        print(f"Agent ID: {agent.id}")
        if hasattr(agent, 'alpha'): print(f"  Alpha: {agent.alpha:.2f}")
        if hasattr(agent, 'gamma'): print(f"  Gamma: {agent.gamma:.3f}")
        if hasattr(agent, 'epsilon_start'):
            print(f"  Epsilon Start: {agent.epsilon_start:.2f}, Min: {agent.epsilon_min:.2f}, Decay: {agent.epsilon_decay_strategy}")

        env = me.get_minihack_envirnment(
            test_env_id,
            observation_keys=['chars', 'blstats', 'pixel'], # Ensure all needed keys are present
            penalty_step=step_penalty,
            penalty_death=death_penalty,
            reward_win=goal_reward,
            max_episode_steps=test_max_episode_steps
        )

        # Instantiate AbstractRLTask correctly
        task = commons.AbstractRLTask(env=env, agent=agent)

        # Setup agent for this experimental run (learning enabled)
        original_learning_state = agent.learning # Should be True by default from agent init
        original_epsilon = agent.epsilon # Should be epsilon_start by default
        
        agent.learning = True # Ensure learning is enabled for this experimental run
        # Epsilon will start at agent.epsilon_start and decay as per its settings during task.interact()
        # No need to set agent.epsilon here, it will use its initial value.

        print(f"  Experimental Run Mode: Learning ENABLED, Epsilon will start at {agent.epsilon:.4f} and decay.")

        # Call interact as defined in commons.AbstractRLTask
        # It returns a list of G_hat_k (average return up to episode k)
        avg_returns_over_time = task.interact(
            n_episodes=test_num_episodes,
            visualize_training_episodes=False # Typically no visualization in automated tests
        )
        
        # Restore agent's state
        agent.learning = original_learning_state
        agent.epsilon = original_epsilon
        env.close() # Close the environment for this agent test

        # Derive episode_total_returns from avg_returns_over_time
        episode_total_returns = []
        current_sum_of_returns = 0.0
        if avg_returns_over_time:
            for k, G_hat_k in enumerate(avg_returns_over_time):
                sum_up_to_k = (k + 1) * G_hat_k
                G_k = sum_up_to_k - current_sum_of_returns
                episode_total_returns.append(G_k)
                current_sum_of_returns = sum_up_to_k
        
        num_successful_episodes = 0
        if episode_total_returns:
            for total_reward in episode_total_returns:
                if total_reward > success_reward_threshold:
                    num_successful_episodes += 1
        
        success_rate = num_successful_episodes / test_num_episodes if test_num_episodes > 0 else 0.0
        
        # Note: avg_steps_to_success cannot be calculated as AbstractRLTask.interact doesn't return per-episode steps.
        # The task.interact method prints average steps, which can be monitored from console.
        
        q_table_size = len(agent.q_table)
        
        results[name] = {
            "success_rate": success_rate,
            "avg_total_reward_final_episode_block": avg_returns_over_time[-1] if avg_returns_over_time else float('nan'),
            "final_epsilon_during_eval": agent.epsilon_min, # Epsilon used during this test run
            "q_table_size": q_table_size
        }
        print(f"\n{name} Results on {test_env_id_str}:")
        print(f"  Success rate (reward > {success_reward_threshold}): {success_rate*100:.1f}% ({num_successful_episodes}/{test_num_episodes})")
        print(f"  Avg total reward (final block avg): {results[name]['avg_total_reward_final_episode_block']:.2f}")
        print(f"  Epsilon during evaluation: {results[name]['final_epsilon_during_eval']:.4f}")
        print(f"  Q-table size: {q_table_size} states")
        print(f"  Note: Per-episode step counts are not available for avg_steps_to_success calculation.")

    print("\n\n=== AGENT TEST SUMMARY (Aligned with AbstractRLTask) ===")
    for name, res in results.items():
        print(f"  {name}: {res['success_rate']*100:.1f}% success, Q-size {res['q_table_size']}, Final Avg Reward {res['avg_total_reward_final_episode_block']:.2f}, Eval Eps {res['final_epsilon_during_eval']:.4f}")
    
    print("\nTest script finished.")

if __name__ == "__main__":
    test_all_agents() 