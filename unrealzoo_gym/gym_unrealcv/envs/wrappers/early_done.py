from gymnasium import Wrapper
import time

class EarlyDoneWrapper(Wrapper):
    def __init__(self, env, max_lost_steps=50):
        super().__init__(env)
        self.max_lost_steps = max_lost_steps

    def step(self, action):
        obs,reward,termination,truncation,info = self.env.step(action) # take a step in the wrapped environment

        if  not info['metrics']['target_viewed']:
            self.count_lost += 1
        else:
            self.count_lost = 0
        env = self.env.unwrapped
        if self.count_lost > self.max_lost_steps:
            info['Truncation'] = True
            truncation = True
        return obs,reward,termination,truncation,info # return the same results as the wrapped environment

    def reset(self, **kwargs):
        states, info = self.env.reset(**kwargs)
        self.start_time = time.time()
        self.count_lost = 0
        return states, info # return the same results as the wrapped environment
