import gym
import numpy as np
from gym.spaces import Box

class ContinuousBanditEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.n_agents = 2   # 2 agents
        self.mean_rewards = np.random.uniform()

        # 输出一个5维的状态 【1， 1， 1， 1， 1】
        self.observation_space = [Box(-1, 1, shape=(5,)) for _ in range(self.n_agents)]
        # 输出联合动作 [0.1, 0.5]
        self.action_space = tuple([Box(-1, 1, shape=(1,))  for a in range(self.n_agents)])
        self.env_info = {"state_shape": self.observation_space[0].shape[0],
                        "obs_shape": self.observation_space[0].shape[0],
                        "n_actions": self.action_space[0].shape[0],
                        "n_agents": self.n_agents,}

    def step(self, action):
        assert len(action) == int(2)
        reward = action[0].item()*action[1].item()
        rewards = np.array(reward).item()
        done = True
        next_state = np.array([1., 1., 1., 1., 1.])
        return next_state, rewards, done, {}

    def reset(self):
        state_init = np.array([1., 1., 1., 1., 1.])
        return state_init
    
