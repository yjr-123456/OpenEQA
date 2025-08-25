import argparse
import gym_unrealcv
import gym
from gym import wrappers
import cv2
import time
import os
import numpy as np
class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealTrack-SuburbNeighborhood_Day-ContinuousMask-v0',
                        help='Select the environment to run')
    parser.add_argument("-r", "--render", default=False, metavar='G', help='show env using cv2')
    args = parser.parse_args()
    env = gym.make(args.env_id)
    env = env.unwrapped
    agent_0 = RandomAgent(env.action_space[0])
    agent_1 = RandomAgent(env.action_space[1])

    episode_count = 100
    rewards = 0
    done = False

    for i in range(episode_count):
        env.seed(i)
        obs = env.reset()
        count_step = 0
        t0 = time.time()
        while True:
            action_0 = agent_0.act(obs, rewards, done)
            action_1 = agent_0.act(obs, rewards, done)
            action_0 = 0
            action_1 = 6
            # action_1 = agent_1.act(obs, rewards, done)
            obs, rewards, done, info = env.step([action_0, action_1])


            count_step += 1

            cv2.imshow('show', obs)
            cv2.waitKey(1)
            if done:
                fps = count_step / (time.time() - t0)
                print ('Fps:' + str(fps))
                break

    # Close the env and write monitor result info to disk
    env.close()


