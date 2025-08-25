import gym
from gym import Wrapper
import time

class TimeDilationWrapper(Wrapper):
    def __init__(self, env, reference_fps=10, update_steps=10, update_dilation=True):
        super().__init__(env)
        self.dilation_factor = 1  # the factor by which to slow down the environment
        self.reference_fps = reference_fps  # the fps at which the environment should run
        self.update_steps = update_steps  # the number of steps after which to update the time dilation
        self.update_dilation = update_dilation  # whether to update the time dilation or not
    def step(self, action):
        obs,reward,termination,truncation,info = self.env.step(action)  # take a step in the wrapped environment
        self.count_steps += 1
        if self.count_steps % self.update_steps == 0:  # update the time dilation every 10 steps
            fps = self.count_steps / (time.time() - self.start_time)
            dilation_factor_new = fps / self.reference_fps  # reference fps: 10
            # print(f'FPS: {fps}', f'Dilation factor: {dilation_factor_new}')
            if  dilation_factor_new / self.dilation_factor > 1.1 or dilation_factor_new / self.dilation_factor < 0.9:
                env = self.env.unwrapped
                if self.update_dilation:
                    self.dilation_factor = dilation_factor_new
                    env.unrealcv.set_global_time_dilation(self.dilation_factor)
        return obs,reward,termination,truncation,info  # return the same results as the wrapped environment

    def reset(self, **kwargs):
        states, info = self.env.reset(**kwargs)
        self.start_time = time.time()
        self.count_steps = 0
        return states, info # return the same results as the wrapped environment