import commons
import numpy as np
import random
from collections import defaultdict

# Import config parameters
from config import MONTE_CARLO_AGENT_PARAMS, SARSA_AGENT_PARAMS, Q_LEARNING_AGENT_PARAMS, DYNA_Q_AGENT_PARAMS
# --- DEBUG IMPORT ---
from config import UNIVERSAL_REWARD_PARAMS # For debug print conditions

class RandomAgent(commons.AbstractAgent):
    """
    Random agent that selects actions randomly from the action space.
    """
    def __init__(self, action_space, params=None):
        """
        A simple agent that takes random actions.

        :param action_space: The action space of the environment 
        """
        super().__init__(id="RandomAgent", action_space=action_space, params=params)
    
    def act(self, state, reward=0):
        """
        Selects a random action from the action space.

        :param state: The current state of the environment (unused by this random agent).
        :param reward: The reward from the previous action (unused by this random agent).
        :return: A randomly selected action.
        """
        return self.action_space.sample()
    
    def learn(self, state, action, reward, next_state, next_action_on_policy, done, truncated):
        pass # Non-learning agent

    def onEpisodeEnd(self, episode_num=None, total_episodes=None):
        pass # Non-learning agent

class FixedAgent(commons.AbstractAgent):
    """
    A fixed agent that always tries to go South until it can no longer move South,
    then it always tries to go East.
    It uses the 'blstats' (x,y coordinates) from the observation to detect
    if it's stuck moving South.
    """
    def __init__(self, action_space, params=None):
        super().__init__(id="FixedAgent", action_space=action_space, params=params)
        self.ACTION_SOUTH = 2
        self.ACTION_EAST = 1
        self.trying_to_move_down = True
        self.previous_y_coord = None # Stores the y-coordinate from the *previous* observation
                                     # to detect if a "South" move was successful.

    def act(self, observation, reward=0):
        current_y_coord = observation['blstats'][1] # y-coordinate is at index 1 of blstats

        action_to_take = None

        if self.trying_to_move_down:
            if self.previous_y_coord is not None and current_y_coord == self.previous_y_coord:
                self.trying_to_move_down = False

            self.previous_y_coord = current_y_coord

            if self.trying_to_move_down: # Check again, as it might have been set to False above
                action_to_take = self.ACTION_SOUTH
            else: # Switched to East in this very step
                action_to_take = self.ACTION_EAST
        else: # Already in the "move East" phase
            action_to_take = self.ACTION_EAST
            # No need to update previous_y_coord if we are only moving east.
            
        return action_to_take

    def learn(self, state, action, reward, next_state, next_action_on_policy, done, truncated):
        pass # Non-learning agent

    def onEpisodeEnd(self, episode_num=None, total_episodes=None):
        """
        Reset agent's internal state for the next episode.
        """
        self.trying_to_move_down = True
        self.previous_y_coord = None
        super().onEpisodeEnd(episode_num=episode_num, total_episodes=total_episodes) # For epsilon decay if ever used

# --- Monte Carlo Agent --- #
class MonteCarloAgent(commons.AbstractAgent):
    def __init__(self, action_space, params=MONTE_CARLO_AGENT_PARAMS):
        effective_params = params if params is not None else MONTE_CARLO_AGENT_PARAMS
        super().__init__(id=effective_params.get("agent_id", "MonteCarloAgent"), action_space=action_space, params=effective_params)
        self.first_visit = self.params.get("first_visit", True)
        self.episode_history = []
        self.returns_sum = defaultdict(lambda: defaultdict(float))
        self.returns_count = defaultdict(lambda: defaultdict(int))

    def _state_to_key(self, state_obs):
        """
        Use chars array to find agent position. Pure positional state - no goal info.
        Fallback to blstats needed for death states where '@' disappears.
        """
        chars = state_obs['chars']
        
        # Find agent position in chars array
        agent_positions = np.where(chars == ord('@'))
        
        if len(agent_positions[0]) > 0:
            # Normal case: agent exists in chars
            agent_x = int(agent_positions[0][0])
            agent_y = int(agent_positions[1][0])
            return (agent_x, agent_y)
        else:
            # Death state: agent '@' has disappeared, use last known position from blstats
            return (int(state_obs['blstats'][0]), int(state_obs['blstats'][1]))

    def act(self, state_obs):
        state_key = self._state_to_key(state_obs)
        self._ensure_q_state_exists(state_key)

        if random.random() < self.epsilon:
            return self.action_space.sample()  # Explore
        else:
            # Exploit: choose action with max Q-value for current state
            q_values_for_state = self.q_table.get(state_key, {})
            if not q_values_for_state: # Should not happen if _ensure_q_state_exists works
                return self.action_space.sample() 
            
            max_q = -float('inf')
            best_actions = []
            for action in range(self.action_space.n):
                q_val = q_values_for_state.get(action, self.q_table_default_value)
                if q_val > max_q:
                    max_q = q_val
                    best_actions = [action]
                elif q_val == max_q:
                    best_actions.append(action)
            return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, next_action_on_policy, done, truncated):
        """For Monte Carlo, 'learn' primarily means storing the experience.
           Actual Q-value updates happen in onEpisodeEnd.
        """
        if not self.learning:
            return
        state_key = self._state_to_key(state)
        self.episode_history.append((state_key, action, reward))

    def onEpisodeEnd(self, episode_num=None, total_episodes=None):
        if not self.learning or not self.episode_history:
            super().onEpisodeEnd(episode_num=episode_num, total_episodes=total_episodes) # Epsilon decay
            self.episode_history = [] # Clear history even if not learning from it this time
            return

        G = 0 # Cumulative discounted return
        step_count = 0
        visited_state_actions = set() # For first-visit MC

        # Iterate backwards through the episode history
        for i in range(len(self.episode_history) - 1, -1, -1):
            state_key, action, reward = self.episode_history[i]
            G = self.gamma * G + reward
            step_count += 1
            if self.first_visit:
                if (state_key, action) not in visited_state_actions:
                    self.returns_sum[state_key][action] += G
                    self.returns_count[state_key][action] += 1
                    visited_state_actions.add((state_key, action))
            else: # Every-visit MC
                self.returns_sum[state_key][action] += G
                self.returns_count[state_key][action] += 1
            
            # Update the main q_table used for acting
            # This Q-value is the average of returns seen so far
            if self.returns_count[state_key][action] > 0: # Avoid division by zero
                 self._set_q_value(state_key, action, self.returns_sum[state_key][action] / self.returns_count[state_key][action])
            else:
                 self._set_q_value(state_key, action, self.q_table_default_value) # Should not happen if count incremented

        self.episode_history = [] # Clear history for the next episode
        super().onEpisodeEnd(episode_num=episode_num, total_episodes=total_episodes) # Handle epsilon decay

# --- SARSA Agent --- #
class SARSAAgent(commons.AbstractAgent):
    def __init__(self, action_space, params=SARSA_AGENT_PARAMS):
        effective_params = params if params is not None else SARSA_AGENT_PARAMS
        super().__init__(id=effective_params.get("agent_id", "SARSAAgent"), action_space=action_space, params=effective_params)
        self.alpha = self.params.get("alpha", 0.1) # Learning rate
        self.step_count = 0
        self.last_updated_q = None # Initialize tracker for the last Q-value updated

    def _state_to_key(self, state_obs):
        """
        Use chars array to find agent position. Pure positional state - no goal info.
        """
        chars = state_obs['chars']
        
        # Find agent position in chars array
        agent_positions = np.where(chars == ord('@'))
        
        if len(agent_positions[0]) > 0:
            # Normal case: agent exists in chars
            agent_x = int(agent_positions[0][0])
            agent_y = int(agent_positions[1][0])
            return (agent_x, agent_y)
        else:
            # Death state: agent '@' has disappeared, use last known position from blstats
            return (int(state_obs['blstats'][0]), int(state_obs['blstats'][1]))

    def act(self, state_obs):
        state_key = self._state_to_key(state_obs)
        self._ensure_q_state_exists(state_key)

        self.step_count += 1

        if random.random() < self.epsilon:
            return self.action_space.sample()  # Epsilon Greedy - Epsilon won so we explore
        else:
            # Exploit: choose action with max Q-value for current state
            q_values_for_state = self.q_table.get(state_key, {})
            if not q_values_for_state: # Should not happen if _ensure_q_state_exists works
                return self.action_space.sample()
            
            max_q = -float('inf')
            best_actions = []
            # Iterate through all possible actions in the action space
            for action_idx in range(self.action_space.n):
                q_val = q_values_for_state.get(action_idx, self.q_table_default_value)
                if q_val > max_q:
                    max_q = q_val
                    best_actions = [action_idx]
                elif q_val == max_q:
                    best_actions.append(action_idx)
            if not best_actions: # Fallback if all Q-values are -inf or state not found (should be rare)
                return self.action_space.sample()
            return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, next_action_on_policy, done, truncated):
        if not self.learning:
            return

        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        current_q = self._get_q_value(state_key, action)
        
        # next_action_on_policy is a_prime for SARSA
        # If done or truncated, the value of the next state Q(s',a') is 0 because the episode has ended.
        if done or truncated:
            next_q_val_for_next_action = 0.0
        else:
            # Ensure the next state_key also has its Q-values initialized if it's new
            self._ensure_q_state_exists(next_state_key)
            next_q_val_for_next_action = self._get_q_value(next_state_key, next_action_on_policy)
        
        # SARSA update rule: Q(s,a) <- Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))
        td_target = reward + self.gamma * next_q_val_for_next_action
        new_q = current_q + self.alpha * (td_target - current_q)
        
        self._set_q_value(state_key, action, new_q)
        self.last_updated_q = new_q # Store the last updated Q-value
        

    def onEpisodeEnd(self, episode_num=None, total_episodes=None):
        super().onEpisodeEnd(episode_num=episode_num, total_episodes=total_episodes) # Handle epsilon decay

        current_episode_display = episode_num + 1 if episode_num is not None else "N/A"
        last_q_display = f"{self.last_updated_q:.4f}" if self.last_updated_q is not None else "N/A"
        
        self.step_count = 0 # Reset for the next episode
        self.last_updated_q = None # Reset for next episode, or it will persist if an episode has 0 learning steps
        pass

# --- Q-Learning Agent --- #
class QLearningAgent(commons.AbstractAgent):
    def __init__(self, action_space, params=Q_LEARNING_AGENT_PARAMS):
        effective_params = params if params is not None else Q_LEARNING_AGENT_PARAMS
        super().__init__(id=effective_params.get("agent_id", "QLearningAgent"), action_space=action_space, params=effective_params)
        self.alpha = self.params.get("alpha", 0.1) # Learning rate
        self.step_count = 0
        self.last_updated_q = None # Initialize tracker for the last Q-value updated
        # --- DEBUG ---
        # Assuming a 5x5 empty room, typical start state is (1,1) after MiniHack's padding.
        # Adjust if your specific "empty_room" variant has a different guaranteed start.
        self.DEBUG_START_STATE = (1,1) 
        self.GOAL_REWARD_VALUE = UNIVERSAL_REWARD_PARAMS.get("goal_reward", 5000) # Get goal reward for debug condition

    def _state_to_key(self, state_obs):
        """
        Use chars array to find agent position. Pure positional state - no goal info.
        Fallback to blstats needed for death states where '@' disappears.
        """
        chars = state_obs['chars']
        
        # Find agent position in chars array
        agent_positions = np.where(chars == ord('@'))
        
        if len(agent_positions[0]) > 0:
            # Normal case: agent exists in chars
            agent_x = int(agent_positions[0][0])
            agent_y = int(agent_positions[1][0])
            return (agent_x, agent_y)
        else:
            # Death state: agent '@' has disappeared, use last known position from blstats
            return (int(state_obs['blstats'][0]), int(state_obs['blstats'][1]))

    def act(self, state_obs):
        state_key = self._state_to_key(state_obs)
        self._ensure_q_state_exists(state_key)

        self.step_count += 1

        if random.random() < self.epsilon:
            action = self.action_space.sample()
            return action
        else:
            # Exploit: choose action with max Q-value for current state
            q_values_for_state = self.q_table.get(state_key, {}) # Already ensures state_key exists
            
            max_q = -float('inf')
            best_actions = []
            # Iterate through all possible actions in the action space
            for action_idx in range(self.action_space.n):
                q_val = q_values_for_state.get(action_idx, self.q_table_default_value)
                if q_val > max_q:
                    max_q = q_val
                    best_actions = [action_idx]
                elif q_val == max_q:
                    best_actions.append(action_idx)
            
            if not best_actions: # Fallback if all Q-values are -inf or state not found
                chosen_action = self.action_space.sample()

            else:
                chosen_action = random.choice(best_actions)
            return chosen_action

    def learn(self, state, action, reward, next_state, next_action_on_policy, done, truncated):
        if not self.learning:
            return

        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        current_q = self._get_q_value(state_key, action)
        
        # Determine the value of max Q(s',a') for the next state s'
        actual_max_next_q_val: float
        if done or truncated:
            actual_max_next_q_val = 0.0  # For terminal states, future rewards are 0
        else:
            self._ensure_q_state_exists(next_state_key) 
            q_values_for_next_state = self.q_table.get(next_state_key, {})
            
            if not q_values_for_next_state: # Should be rare if _ensure_q_state_exists populates all actions
                actual_max_next_q_val = self.q_table_default_value # Fallback to default for new/empty next states
            else:
                actual_max_next_q_val = -float('inf') # Start with negative infinity to find the true max
                for next_action_candidate in range(self.action_space.n): # Iterate over all possible actions
                    q_val = q_values_for_next_state.get(next_action_candidate, self.q_table_default_value)
                    if q_val > actual_max_next_q_val:
                        actual_max_next_q_val = q_val
            
        # Q-learning update rule: Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
        td_target = reward + self.gamma * actual_max_next_q_val
        # td_error = td_target - current_q # This is the TD error itself
        new_q = current_q + self.alpha * (td_target - current_q) # Applying the TD error
        
        self._set_q_value(state_key, action, new_q)
        self.last_updated_q = new_q


    def onEpisodeEnd(self, episode_num=None, total_episodes=None):
        super().onEpisodeEnd(episode_num=episode_num, total_episodes=total_episodes) # Handle epsilon decay

        current_episode_display = episode_num + 1 if episode_num is not None else "N/A"
        last_q_display = f"{self.last_updated_q:.4f}" if self.last_updated_q is not None else "N/A"
        print(f"Q-Value of last episode is: {last_q_display}")

        self.step_count = 0 
        self.last_updated_q = None
        pass

# --- Dyna-Q Agent --- #
class DynaQAgent(QLearningAgent): # Inherits from QLearningAgent
    def __init__(self, action_space, params=DYNA_Q_AGENT_PARAMS):
        effective_params = params if params is not None else DYNA_Q_AGENT_PARAMS
        # Call QLearningAgent's (super) init.
        # QLearningAgent's __init__ will handle its own `params` assignment correctly with this change.
        # We pass `effective_params` here, which QLearningAgent will then use.
        super().__init__(action_space, params=effective_params)
        self.id = effective_params.get("agent_id", "DynaQAgent") # explicit_id can be set if needed by DynaQ
        self.planning_steps = self.params.get("planning_steps", 10)  # Uses self.params set by super()
        self.model = {}  # Stores (state, action) -> (reward, next_state, done_sim)
        # We need to store 'done_sim' in the model because the Q-learning update for simulated steps
        # needs to know if the simulated next_state is terminal according to the model.

    def learn(self, state, action, reward, next_state, next_action_on_policy, done, truncated):
        # 1. Learn from real experience (standard Q-learning update)
        super().learn(state, action, reward, next_state, next_action_on_policy, done, truncated)

        if not self.learning: # if agent is not in learning mode, skip model update and planning
            return

        # 2. Update the model
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        if state_key not in self.model:
            self.model[state_key] = {}
        self.model[state_key][action] = (reward, next_state_key, done) # Store 'done' from real experience

        # 3. Perform planning steps (learning from the model)
        if not self.model: # No experiences in model yet
            return
            
        for _ in range(self.planning_steps):
            # Randomly sample a previously experienced state and action
            s_key_sim = random.choice(list(self.model.keys()))
            if not self.model[s_key_sim]: # If no actions recorded for this state_key in model
                continue 
            a_sim = random.choice(list(self.model[s_key_sim].keys()))
            
            r_sim, next_s_key_sim, done_sim = self.model[s_key_sim][a_sim]
            
            current_q_sim = self._get_q_value(s_key_sim, a_sim)
            
            max_next_q_val_sim = self.q_table_default_value
            if not done_sim: # If the simulated next state is not terminal
                self._ensure_q_state_exists(next_s_key_sim)
                q_values_for_next_state_sim = self.q_table.get(next_s_key_sim, {})
                if q_values_for_next_state_sim:
                    max_next_q_val_sim = -float('inf')
                    for next_action_candidate_sim in range(self.action_space.n):
                        q_val_sim = q_values_for_next_state_sim.get(next_action_candidate_sim, self.q_table_default_value)
                        if q_val_sim > max_next_q_val_sim:
                            max_next_q_val_sim = q_val_sim
                # else max_next_q_val_sim remains self.q_table_default_value
            
            td_target_sim = r_sim + self.gamma * max_next_q_val_sim
            new_q_sim = current_q_sim + self.alpha * (td_target_sim - current_q_sim)
            self._set_q_value(s_key_sim, a_sim, new_q_sim)

    def onEpisodeEnd(self, episode_num=None, total_episodes=None):
        current_episode_display = episode_num + 1 if episode_num is not None else "N/A"
        super().onEpisodeEnd(episode_num=episode_num, total_episodes=total_episodes)
