import gym
from gym import Wrapper
import json

class LoadNavConfigWrapper(Wrapper):
    def __init__(self, env, nav_config_path=None):
        if nav_config_path is not None:
            with open(nav_config_path, 'r') as f:
                self.nav_config = json.load(f)
                env.unwrapped.nav_configs = self.nav_config
        else:
            self.nav_config = None
        super().__init__(env)
    def step(self, action):
        obs,reward,termination,truncation,info = self.env.step(action)
        return obs,reward,termination,truncation,info

    def reset(self, **kwargs):
        states,info = self.env.reset(**kwargs)
        return states,info