import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
import minihack_env as me
import matplotlib.pyplot as plt
import commons
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import os # Import os module for directory creation


class GridWorldEnv(gym.Env):
    """
    My demo Grid World environment where an agent needs to navigate from (0,0) to (n-1,m-1).
    """
    def __init__(self, n=5 , m=5):
        """
        Initialize the Grid World environment.
        """
        super(GridWorldEnv, self).__init__()
        
        # Environment dimensions
        self.n = n
        self.m = m
        
        # Define action space (4 possible actions)
        self.action_space = spaces.Discrete(4)
        
        # Define action sets
        self.action_set = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Right
            2: (0, 1),   # Down
            3: (0, -1)   # Left
        }
        
        # Define observation space (2D coordinates)
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([n-1, m-1]),
            dtype=np.int32
        )
        
        # Initialize state
        self.state = None

        # Initialize goalstate
        self.goal_state = np.array([n-1, m-1])

    def reward_function(self, state):
        if np.array_equal(state, self.goal_state):
            return 0
        else:
            return -1
        
    def reset(self, seed=None):
        """
        Reset my environment to initial state.
        """
        super().reset(seed=seed)
        self.state = np.array([0, 0])  # Start at (0,0)
        return self.state, {}
        
    def step(self, action):
        """
        Take a step in our environment.
        """
        # Get the movement from action
        dx, dy = self.action_set[action]
        
        # Calculate new position
        new_x = self.state[0] + dx
        new_y = self.state[1] + dy
        
        # Check if the new position is valid
        if 0 <= new_x < self.n and 0 <= new_y < self.m:
            self.state = np.array([new_x, new_y])
        
        # Check if goal is reached
        done = bool(np.array_equal(self.state, self.goal_state))
        
        # Calculate reward
        reward = self.reward_function(self.state)
        
        truncated = False
        return self.state, reward, done, truncated, {}

    def render(self):
        """
        Render the current state of the environment.
        
        Returns:
            str: String representation of the grid
        """
        # Create empty grid
        grid = [['.' for _ in range(self.m)] for _ in range(self.n)]
        
        # Mark the goal
        grid[self.goal_state[0]][self.goal_state[1]] = '$'
        
        # Mark the agent
        grid[self.state[0]][self.state[1]] = '@'
        
        # Convert to string
        return '\n'.join([' '.join(row) for row in grid])

    def close(self):
        # Close the environment
        pass

# Test the environment with RLTask
if __name__ == "__main__":
    # Create environment
    env = GridWorldEnv()
    
    # Create random agent
    from agents import RandomAgent
    agent = RandomAgent(env.action_space)
    
    # Create RL task using AbstractRLTask from commons, with correct argument order
    task = commons.AbstractRLTask(env, agent)
    
    # Run for 10000 episodes and plot average returns
    num_episodes = 10000
    print(f"\nRunning {num_episodes} episodes with RandomAgent...")
    avg_returns = task.interact(num_episodes)
    
    # Plot the evolution of average returns
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_episodes + 1), avg_returns)
    plt.title('Evolution of Average Returns over 10000 Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Average Return')
    plt.grid(True)
        
    # Define the directory and ensure it exists
    output_dir = "results/1/"
    os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist

    # Save the plot to the specified directory
    file_path = os.path.join(output_dir, "average_returns.png")
    plt.savefig(file_path)
    print(f"\nAverage returns plot saved to {file_path}")
    #plt.show()

    print("\nVisualizing first 10 steps of a new episode:")
    task.visualize_episode(max_number_steps=10)
    env.close() # Call close on the environment if needed


