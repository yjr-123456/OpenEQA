import gym
from gym import Wrapper
import json

class ConfigUEWrapper(Wrapper):
    def __init__(self, env, height_bias=0):
        super().__init__(env)
        self.height_bias = height_bias

    def step(self, action):
        obs,reward,termination,truncation,info = self.env.step(action)
        return obs,reward,termination,truncation,info

    def reset(self, **kwargs):
        env = self.env.unwrapped
        
        states,info = self.env.reset(**kwargs)
        return states, info