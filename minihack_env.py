from minihack.envs.room import MiniHackRoom
import gymnasium as gym
import minihack
from nle import nethack
from minihack import RewardManager
from config import UNIVERSAL_REWARD_PARAMS


ACTIONS = tuple(nethack.CompassCardinalDirection)
EMPTY_ROOM = "empty-room"
ROOM_WITH_LAVA = "room-with-lava"
ROOM_WITH_MONSTER = "room-with-monster"
ROOM_WITH_MULTIPLE_MONSTERS = "room-with-multiple-monsters"
CLIFF = "cliff-minihack"




des_room_with_lava ="""
MAZE: "mylevel", ' '
FLAGS:premapped
GEOMETRY:center,center
MAP
|-----     ------
|.....-- --.....|
|.T.T...-.....L.|
|...T...........|
|.T.T...-.....L.|
|.....-----.....|
|-----     ------
ENDMAP
"""




des_cliff="""
MAZE: "mylevel", ' '
FLAGS:premapped
GEOMETRY:center,center
MAP
|----------------
|...............|
|...............|
|...............|
|...............|
|.LLLLLLLLLLLLL.|
|----------------
ENDMAP
"""


des_monster = """
MAZE: "mylevel", ' '
FLAGS:premapped
GEOMETRY:center,center
MAP
.....
.....
.....
.....
.....
ENDMAP
REGION: (0,0,20,80), lit, "ordinary"
MONSTER: ('Z', "ghoul"), (2,4)
"""


"""Reshaping rewards with death levels"""
class DoNotResetWhenDead(gym.Wrapper):

    def __init__(self, env, 
                 max_episode_steps = 1000, 
                 goal_reward = UNIVERSAL_REWARD_PARAMS["goal_reward"], 
                 negative_step_reward = UNIVERSAL_REWARD_PARAMS["step_penalty"], 
                 dead_negative_reward = UNIVERSAL_REWARD_PARAMS["death_penalty"]):
        super().__init__(env)
        self.step_counter = 0
        self.max_episode_steps = max_episode_steps
        self.goal_reward = goal_reward
        self.negative_step_reward = negative_step_reward
        self.dead_negative_reward = dead_negative_reward

    def step(self, action, **kwargs):
        # We override the done
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.step_counter += 1
        if isinstance(self.env, MiniHackRoom):
            env = self.env.unwrapped
        else:
            env = self.env.env.unwrapped
        
        if info["end_status"] == env.StepStatus.ABORTED or self.step_counter == self.max_episode_steps:
            terminated = truncated = True
            reward = self.negative_step_reward
        elif info["end_status"] == env.StepStatus.DEATH:
            terminated = truncated = False
            reward = self.dead_negative_reward  + self.negative_step_reward
            observation, info = self.env.reset(**kwargs)
        elif info["end_status"] == env.StepStatus.TASK_SUCCESSFUL:
            terminated = truncated  = True
            reward = self.goal_reward

        else:
            terminated = truncated = False
            reward = self.negative_step_reward
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.step_counter = 0
        return observation, info



def get_minihack_envirnment(id, **kwargs):

    size = kwargs.get("size", 5)
    random = kwargs.get("random", False)
    add_pixels = kwargs.get("add_pixel", False)
    max_episode_steps = kwargs.get("max_episode_steps", 1000)

    # Fetch universal reward params
    goal_reward = UNIVERSAL_REWARD_PARAMS["goal_reward"]
    step_penalty = UNIVERSAL_REWARD_PARAMS["step_penalty"]
    death_penalty = UNIVERSAL_REWARD_PARAMS["death_penalty"]

    if "observation_keys" in kwargs:
        obs = kwargs["observation_keys"]
    elif add_pixels:
        obs = ["chars", "pixel"]
    else:
        obs = ["chars"]

    if id == EMPTY_ROOM:
        env = MiniHackRoom(size = size,
                           max_episode_steps=50,
                           actions=ACTIONS,
                           random = random,
                           observation_keys=obs,
                           reward_win=goal_reward,
                           penalty_step = step_penalty,
                           penalty_time=step_penalty,
                           reward_lose=death_penalty)
    elif id == ROOM_WITH_LAVA:
        des_file = des_room_with_lava
        if not random:
            cont = """STAIR:(15,3),down \nBRANCH: (3,3,3,3),(4,4,4,4) \n"""
            des_file += cont

        env = gym.make(
            "MiniHack-Navigation-Custom-v0",
            actions=ACTIONS,
            des_file=des_file,
            max_episode_steps=max_episode_steps,
            observation_keys=obs
        )
        try:
            nle_env = env.unwrapped
            if hasattr(nle_env, "_max_episode_steps"):
                nle_env._max_episode_steps = max_episode_steps
        except Exception as e:
            print(f"[get_minihack_environment] WARNING: ROOM_WITH_LAVA: Could not set _max_episode_steps on unwrapped env: {e}")
        env = DoNotResetWhenDead(env, max_episode_steps, 
                                 goal_reward=goal_reward,
                                 negative_step_reward=step_penalty,
                                 dead_negative_reward=death_penalty)
    elif id == CLIFF:
        des_file = des_cliff
        if not random:
            cont = """STAIR:(15,5),down \nBRANCH: (1,5,1,5),(4,4,4,4) \n"""
            des_file += cont

        env = gym.make(
            "MiniHack-Navigation-Custom-v0",
            actions=ACTIONS,
            des_file=des_file,
            max_episode_steps=max_episode_steps,
            observation_keys=obs
        )
        try:
            nle_env = env.unwrapped
            if hasattr(nle_env, "_max_episode_steps"):
                nle_env._max_episode_steps = max_episode_steps
        except Exception as e:
            print(f"[get_minihack_environment] WARNING: CLIFF: Could not set _max_episode_steps on unwrapped env: {e}")
        env = DoNotResetWhenDead(env, max_episode_steps,
                                 goal_reward=goal_reward,
                                 negative_step_reward=step_penalty,
                                 dead_negative_reward=death_penalty)
    elif id == ROOM_WITH_MONSTER:
        des_file = des_monster
        if not random:
            cont = """STAIR:(4,4),down \nBRANCH: (0,0,0,0),(1,1,1,1) \n"""
            des_file += cont

        env = gym.make(
            "MiniHack-Navigation-Custom-v0",
            actions=ACTIONS,
            des_file=des_file,
            max_episode_steps=max_episode_steps,
            observation_keys=obs
        )
        try:
            nle_env = env.unwrapped
            if hasattr(nle_env, "_max_episode_steps"):
                nle_env._max_episode_steps = max_episode_steps
        except Exception as e:
            print(f"[get_minihack_environment] WARNING: ROOM_WITH_MONSTER: Could not set _max_episode_steps on unwrapped env: {e}")
        env = DoNotResetWhenDead(env, max_episode_steps,
                                 goal_reward=goal_reward,
                                 negative_step_reward=step_penalty,
                                 dead_negative_reward=death_penalty)
    elif id == ROOM_WITH_MULTIPLE_MONSTERS:
        size = 7
        env = MiniHackRoom(size=size,
                           max_episode_steps=100,
                           actions=ACTIONS,
                           random=random,
                           n_monster=3,
                           observation_keys=obs,
                           reward_win=goal_reward,
                           reward_lose=death_penalty,
                           penalty_step=step_penalty,
                           penalty_time=step_penalty)
    else:
        raise Exception("Environment %s not found" % str(id))

    return env
