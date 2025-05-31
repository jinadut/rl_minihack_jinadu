from typing import List, Any, Tuple, Dict, Optional
import numpy as np
from gymnasium import Env
import matplotlib.pyplot as plt
import time # Added for time.sleep

# Define BLSTATS_INDICES for coordinate extraction
BLSTATS_INDICES = {'X': 0, 'Y': 1}

# Import TRAINING_PARAMS for fallback in interact loop
from config import TRAINING_PARAMS, UNIVERSAL_REWARD_PARAMS


class AbstractAgent():

    def __init__(self, id: str, action_space: Any, params: Optional[Dict[str, Any]] = None):
        """
        An abstract interface for an agent.

        :param id: Unique identifier for the agent.
        :param action_space: Representation of the agent's possible actions (e.g., gym.Env.action_space).
        :param params: Dictionary of parameters for the agent, e.g., from config.py.
        """
        self.id = id
        self.action_space = action_space
        self.learning = True
        self.params = params if params is not None else {}

        # Epsilon-greedy parameters
        self.epsilon_start = self.params.get("epsilon_start", 1.0)
        self.epsilon_min = self.params.get("epsilon_min", 0.01)
        self.epsilon_decay_rate = self.params.get("epsilon_decay_rate", 0.995)
        self.epsilon_decay_strategy = self.params.get("epsilon_decay_strategy", "multiplicative") # Default to old behavior
        self.epsilon = self.epsilon_start 

        # Common parameters for learning agents, initialized from params or defaults
        self.gamma = self.params.get("gamma", 0.99)
        self.q_table = {}
        self.q_table_default_value = self.params.get("q_table_default_value", 0.0)

    def act(self, state_obs: Any, blstats: Optional[np.ndarray] = None) -> Any:
        """
        This function represents the actual decision-making process of the agent. Given a 'state' 
        (which could be an observation dict) and possibly blstats, the agent returns an action.
        Derived classes must implement this.

        Args:
            state_obs (Any): The observation from the environment.
            blstats (Optional[np.ndarray]): The blstats array, for agents using coordinate-based states.

        Returns:
            Any: The action to take.
        """
        raise NotImplementedError()

    def learn(
        self, 
        state: Any, 
        action: Any, 
        reward: float, 
        next_state: Any, 
        next_action_on_policy: Optional[Any], # Specifically for SARSA
        done: bool, 
        truncated: bool
    ) -> None:
        """
        Agent learns from the experience (s, a, r, s', [next_a_on_policy], done, truncated).
        This method will be overridden by specific learning agents.
        Non-learning agents can leave it as a pass.
        """
        if not self.learning:
            return
        pass # Default implementation for non-learning or to be overridden

    def onEpisodeEnd(self, episode_num: Optional[int] = None, total_episodes: Optional[int] = None) -> None:
        """Handles end-of-episode updates, like epsilon decay."""
        if not self.learning: # Don't decay epsilon if not in learning mode
            return

        if self.epsilon_decay_strategy == "linear":
            if total_episodes is not None and total_episodes > 0 and episode_num is not None:
                # Calculate decay_amount and new_epsilon for every episode
                decay_amount = (self.epsilon_start - self.epsilon_min) / total_episodes
                # episode_num is 0-indexed.
                # Epsilon will be epsilon_start for actions taken in episode 0.
                # It will decay linearly for subsequent episodes.
                new_epsilon_val = self.epsilon_start - (episode_num * decay_amount)
                self.epsilon = max(self.epsilon_min, new_epsilon_val)
                
                if episode_num % 100 == 0: # Print/log every 100 episodes to avoid too much spam
                    # Example logging:
                    # print(f"Episode {episode_num + 1}/{total_episodes}: Epsilon decayed to {self.epsilon:.4f}")
                    pass # Placeholder for actual logging if desired
            # If total_episodes or episode_num is not provided, epsilon doesn't change for linear strategy
            # or one might add a warning/error, but for now, it just means no decay in that call.
        elif self.epsilon_decay_strategy == "multiplicative":
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay_rate
                self.epsilon = max(self.epsilon_min, self.epsilon) # Ensure it doesn't go below min
        # If strategy is unknown, epsilon doesn't change
        # print(f"Agent {self.id}: Epsilon at end of episode {episode_num+1 if episode_num is not None else 'N/A'}: {self.epsilon:.4f}")
        return

    def _get_q_value(self, state_key: Any, action: Any) -> float:
        """Helper to get Q-value, handling missing states/actions with default."""
        return self.q_table.get(state_key, {}).get(action, self.q_table_default_value)

    def _set_q_value(self, state_key: Any, action: Any, value: float) -> None:
        """Helper to set Q-value, creating state entry if needed."""
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        self.q_table[state_key][action] = value

    def _ensure_q_state_exists(self, state_key: Any):
        """Ensures a state entry exists in the Q-table, typically with default values for all actions."""
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: self.q_table_default_value for a in range(self.action_space.n)}


def get_state_representation(obs_dict: Dict, blstats: Optional[np.ndarray], state_representation_type: str = "chars_hash") -> Any:
    """
    Generates a state representation based on the specified type.

    Args:
        obs_dict (dict): The observation dictionary from the environment, potentially containing 'chars'.
        blstats (numpy.ndarray): The blstats array from the environment, containing agent coordinates.
                                 Indices for x and y are defined by BLSTATS_INDICES.
        state_representation_type (str): Type of state representation.
                                         "coords" for (x,y) tuple.
                                         "chars_hash" for hash of the 'chars' grid.

    Returns:
        A hashable state representation (tuple for coords, int for chars_hash).
        Returns None if the required data for the chosen representation is missing or an error occurs.
    """
    if state_representation_type == "coords":
        if blstats is not None and len(blstats) > max(BLSTATS_INDICES['X'], BLSTATS_INDICES['Y']):
            try:
                x = int(blstats[BLSTATS_INDICES['X']])
                y = int(blstats[BLSTATS_INDICES['Y']])
                return (x, y)  # Return a tuple, which is hashable
            except (IndexError, ValueError) as e:
                # print(f"[ERROR get_state_representation] Could not extract coordinates from blstats: {e}")
                return None 
        else:
            # print(f"[WARNING get_state_representation] 'coords' type requested, but blstats missing or too short.")
            return None
    
    elif state_representation_type == "chars_hash":
        if "chars" in obs_dict and obs_dict["chars"] is not None:
            try:
                return hash(obs_dict["chars"].tobytes())
            except AttributeError: # If obs_dict["chars"] is not a numpy array or doesn't have tobytes
                # print(f"[ERROR get_state_representation] 'chars' object does not support tobytes(). Type: {type(obs_dict['chars'])}")
                return None
        else:
            # print(f"[WARNING get_state_representation] 'chars_hash' type requested, but 'chars' missing from observation.")
            return None
    
    else:
        # print(f"[WARNING get_state_representation] Unknown state_representation_type: {state_representation_type}.")
        return None


class AbstractRLTask():

    def __init__(self, env: Env, agent: AbstractAgent):
        """
        This class abstracts the concept of an agent interacting with an environment.


        :param env: the environment to interact with (e.g. a gym.Env)
        :param agent: the interacting agent
        """

        self.env = env
        self.agent = agent
        self._viz_fig_train = None # Figure for in-training visualization

    def _setup_train_viz_figure(self, initial_state_pixel):
        if self._viz_fig_train is None or not plt.fignum_exists(self._viz_fig_train.number):
            self._viz_fig_train = plt.figure(figsize=(6,6)) # Use a specific figure
            # plt.show(block=False) # Ensure it's non-blocking if shown immediately
        plt.figure(self._viz_fig_train.number) # Switch to this figure
        plt.clf()
        plt.imshow(initial_state_pixel)
        plt.title("In-Training Visualization")
        plt.axis('off')
        plt.pause(0.1) # Initial pause to render

    def _update_train_viz_figure(self, pixel_data, episode_num, step_num, action, reward):
        if self._viz_fig_train is None or not plt.fignum_exists(self._viz_fig_train.number):
            # If figure was closed, re-initialize (though ideally it stays open)
            self._setup_train_viz_figure(pixel_data)
        else:
            plt.figure(self._viz_fig_train.number)
            plt.clf()
            plt.imshow(pixel_data)
        plt.title(f"Ep: {episode_num+1}, Step: {step_num}, Act: {action}, Rew: {reward:.2f}")
        plt.axis('off')
        plt.draw()
        plt.pause(0.05) # Shorter pause for updates

    def _close_train_viz_figure(self):
        if self._viz_fig_train is not None and plt.fignum_exists(self._viz_fig_train.number):
            plt.close(self._viz_fig_train)
        self._viz_fig_train = None

    def interact(self, n_episodes: int, 
                 visualize_training_episodes: bool = False,
                 visualize_every_n_episodes: int = 100,
                 max_steps_for_viz_in_train: int = 50) -> List[float]:
        """
        This function executes n_episodes of interaction between the agent and the environment.
        It collects total returns for each episode and then computes the average returns
        using self.get_average_return().

        :param n_episodes: the number of episodes of the interaction
        :return: a list of episode average returns
        """
        episode_total_returns = []
        episode_steps_taken = [] # To store steps for each episode
        print(f"Starting interaction for {n_episodes} episodes with agent: {self.agent.id}")
        
        original_agent_learning_state = self.agent.learning

        for episode_num in range(n_episodes):
            do_visualize_this_episode = visualize_training_episodes and \
                                        (episode_num == 0 or (episode_num + 1) % visualize_every_n_episodes == 0)
            
            if do_visualize_this_episode:
                print(f"--- Visualizing Training Episode {episode_num + 1}/{n_episodes} ---")
                self.agent.learning = False             
            reset_val = self.env.reset()
            if isinstance(reset_val, tuple) and len(reset_val) == 2 and isinstance(reset_val[1], dict):
                current_state_obs, info = reset_val
            else:
                current_state_obs = reset_val
                info = {}
            
            done = False
            truncated = False
            current_episode_return = 0.0
            steps_this_episode = 0
            current_action = self.agent.act(current_state_obs)

            if do_visualize_this_episode and isinstance(current_state_obs, dict) and 'pixel' in current_state_obs:
                self._setup_train_viz_figure(current_state_obs['pixel'])

            while not (done or truncated):
                step_result = self.env.step(current_action)
                steps_this_episode += 1

                if len(step_result) == 5:
                    next_state_obs, reward, done, truncated, info_step = step_result
                elif len(step_result) == 4:
                    next_state_obs, reward, done, info_step = step_result
                    # truncated = done
                else:
                    raise ValueError("Unexpected number of values from env.step()")
                
                info.update(info_step if isinstance(info_step, dict) else {})
                current_episode_return += reward
                
                next_action_on_policy = self.agent.act(next_state_obs)
                
                # Perform learning step, even if visualizing (unless learning explicitly disabled for viz)
                self.agent.learn(
                    current_state_obs, current_action, reward, next_state_obs, 
                    next_action_on_policy, done, truncated
                )
                
                if do_visualize_this_episode:
                    print(f"  Ep {episode_num+1}, Step {steps_this_episode}: Action={current_action}, Reward={reward:.2f}")
                    if isinstance(next_state_obs, dict) and 'pixel' in next_state_obs and self._viz_fig_train:
                        self._update_train_viz_figure(next_state_obs['pixel'], episode_num, steps_this_episode, current_action, reward)
                    elif isinstance(next_state_obs, dict) and 'chars' in next_state_obs: # Fallback to text for viz
                        print(self.env.render())
                        time.sleep(0.1)
                    if steps_this_episode >= max_steps_for_viz_in_train:
                        print(f"  Reached max visualization steps ({max_steps_for_viz_in_train}) for this training episode.")
                        # We let the episode continue to its natural end for learning, just stop visualizing steps.
                        # To actually truncate here for viz episodes, would need more logic.
                        pass # Or break if we want to shorten visualized training episodes
                
                current_state_obs = next_state_obs
                current_action = next_action_on_policy

                if do_visualize_this_episode and steps_this_episode >= max_steps_for_viz_in_train:
                     # If we were visualizing, and hit max viz steps, stop visualizing further steps in this episode
                     # but let the episode continue for learning until done/truncated. 
                     # The visualization window remains open until the episode actually ends or is closed manually.
                     pass 

            if do_visualize_this_episode:
                print(f"--- End of Visualized Training Episode {episode_num + 1}. Total Reward: {current_episode_return:.2f} ---")
                # self._close_train_viz_figure() # Keep figure open until next viz episode or end of training
                self.agent.learning = original_agent_learning_state # Restore learning state if it was changed
            
            episode_total_returns.append(current_episode_return)
            episode_steps_taken.append(steps_this_episode)
            self.agent.onEpisodeEnd(episode_num=episode_num, total_episodes=n_episodes)

            if (episode_num + 1) % 10 == 0:
                avg_return_last_100 = np.mean(episode_total_returns[-100:])
                avg_steps_last_100 = np.mean(episode_steps_taken[-100:])
                print(f"Episode {episode_num + 1}/{n_episodes} completed. Steps: {steps_this_episode}. Avg Ret (last 100): {avg_return_last_100:.2f}. Avg Steps (last 100): {avg_steps_last_100:.2f}")
        
        self._close_train_viz_figure() # Close any open in-training viz window at the end of all interactions
        self.agent.learning = original_agent_learning_state # Ensure agent learning state is restored

        average_returns = self.get_average_return(episode_total_returns)
        avg_steps_overall = np.mean(episode_steps_taken) if episode_steps_taken else 0
        print(f"Interaction finished. Final Epsilon for agent {self.agent.id}: {self.agent.epsilon:.4f}")
        print(f"Overall average steps per episode: {avg_steps_overall:.2f}")
        return average_returns


    def visualize_episode(self, max_number_steps = 10):
        """
        This function executes and plot an episode (or a fixed number 'max_number_steps' steps).
        You may want to disable some agent behaviours when visualizing(e.g. self.agent.learning = False)
        If 'pixel' is in the observation, it will be rendered using matplotlib.imshow().
        Otherwise, it will try to print self.env.render().
        :param max_number_steps: Optional, maximum number of steps to plot.
        :return:
        """
        initial_learning_state = self.agent.learning
        self.agent.learning = False # Disable learning during visualization

        step_count = 0
        done = False
        reset_output = self.env.reset()
        if isinstance(reset_output, tuple) and len(reset_output) == 2 and isinstance(reset_output[1], dict):
            state, info = reset_output
        else:
            state = reset_output
            info = {}

        pixel_render_available = isinstance(state, dict) and 'pixel' in state
        fig = None # Initialize fig variable

        if pixel_render_available:
            fig = plt.figure() # Store the figure reference
            plt.imshow(state['pixel'])
            plt.title("Agent Visualization")
            plt.axis('off')
            # plt.show(block=False) # Removed
            plt.pause(0.1) # This should be enough to show the first frame

        while not done:
            if max_number_steps is not None and step_count >= max_number_steps:
                break
            
            action = self.agent.act(state)
            step_result = self.env.step(action)
            
            if len(step_result) == 4:
                next_state, reward, done, info_step = step_result
            elif len(step_result) == 5:
                next_state, reward, terminated, truncated, info_step = step_result
                done = terminated or truncated
            else:
                raise ValueError("Unexpected number of values returned by env.step()")

            state = next_state
            info.update(info_step if isinstance(info_step, dict) else {})
            
            if pixel_render_available and isinstance(state, dict) and 'pixel' in state and fig is not None:
                plt.figure(fig.number) # Ensure we are operating on the correct figure
                plt.clf()
                plt.imshow(state['pixel'])
                plt.title(f"Step: {step_count + 1}")
                plt.axis('off')
                plt.draw()
                plt.pause(0.2) 
            elif hasattr(self.env, 'render'):
                print(f"\nStep: {step_count + 1}")
                render_output = self.env.render()
                if render_output is not None:
                    print(render_output)
                time.sleep(0.1)
            else:
                print(f"Step: {step_count + 1}, No pixel data and env.render() not available/suitable.")
                time.sleep(0.1)

            step_count += 1
        
        if fig is not None: # If a figure was created
            plt.close(fig) # Close the specific figure
        
        self.agent.learning = initial_learning_state # Restore agent's learning state

    
    def get_average_return(self, episode_returns: List[float]) -> List[float]:
        """
        This function returns a list of average returns (G_hat_k) for each episode.
        G_hat_k = (1 / (k + 1)) * sum(G_i for i in 0 to k), where k is the 0-based episode index.
        :param episode_returns: a list of returns [G0, G1, G2, ...]
        :return: a list of average returns [G_hat_0, G_hat_1, G_hat_2, ...]
        """
        average_returns = []
        current_sum_of_returns = 0.0
        for k, G_k in enumerate(episode_returns):  # k is the 0-based index, G_k is the return for episode k
            current_sum_of_returns += G_k
            average_G_k = current_sum_of_returns / (k + 1)
            average_returns.append(average_G_k)
        return average_returns


blank = 32
def get_crop_chars_from_observation(observation):
    chars = observation["chars"]
    coords = np.argwhere(chars != blank)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    chars = chars[x_min:x_max + 1, y_min:y_max + 1]
    return chars


size_pixel = 16
def get_crop_pixel_from_observation(observation):
    coords = np.argwhere(observation["chars"] != blank)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    non_empty_pixels = observation["pixel"][x_min * size_pixel : (x_max + 1) * size_pixel, y_min * size_pixel : (y_max + 1) * size_pixel]
    return non_empty_pixels

import gymnasium as gym
from gymnasium.core import ObservationWrapper
from gymnasium import spaces

class PixelObservationWrapper(ObservationWrapper):
    """Wrapper to extract pixel observations from MiniHack's Dict observation space."""
    def __init__(self, env):
        super().__init__(env)
        # The observation space is now just the pixel part
        self.observation_space = env.observation_space.spaces['pixel']
        # Ensure the observation space is Box and has 3 dimensions (H, W, C)
        # MiniHack 'pixel' is typically (H, W, C) with C=3 (RGB)
        # If it's not, you might need to reshape or transpose.
        # For SB3 CnnPolicy, it expects (C, H, W) by default if data_format='channels_first'
        # or (H, W, C) if data_format='channels_last'. SB3 handles this internally if input is (H,W,C)
        if not isinstance(self.observation_space, spaces.Box):
            raise ValueError(
                f"Expected Box space for pixel observations, got {type(self.observation_space)}"
            )
        if len(self.observation_space.shape) != 3:
            raise ValueError(
                f"Expected 3D shape for pixel (H, W, C), got {self.observation_space.shape}"
            )

    def observation(self, observation):
        """Extracts the pixel observation."""
        return observation['pixel']