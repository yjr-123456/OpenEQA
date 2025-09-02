import argparse
import gym_unrealcv
import gymnasium as gym
from gymnasium import wrappers
import cv2
import time
import numpy as np
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, agents, augmentation, configUE
from gym_unrealcv.envs.tracking.baseline import PoseTracker, Nav2GoalAgent
from Keyboard_NavigationAgent import get_key_action,on_press,on_release
import random
import os
import torch
import math
from pynput import keyboard


os.environ['UnrealEnv']='D:\\UnrealEnv\\'

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealTrack-DowntownWest-ContinuousMask-v0',
                        help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=1, help='random seed')
    parser.add_argument("-t", '--time_dilation', dest='time_dilation', default=30, help='time_dilation to keep fps in simulator')
    parser.add_argument("-d", '--early_done', dest='early_done', default=-1, help='early_done when lost in n steps')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')

    args = parser.parse_args()
    print(args.env_id)
    env = gym.make(args.env_id)
    if int(args.time_dilation) > 0:  # -1 means no time_dilation
        env = time_dilation.TimeDilationWrapper(env, args.time_dilation)
    if int(args.early_done) > 0:  # -1 means no early_done
        env = early_done.EarlyDoneWrapper(env, args.early_done)
    if args.monitor:
        env = monitor.DisplayWrapper(env)

    env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
    env = agents.NavAgents(env, mask_agent=True)
    episode_count = 50
    rewards = 0
    done = False

    Total_rewards = 0
    env.seed(int(args.seed))

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    try:
        for eps in range(0, episode_count):
            obs = env.reset()
            agents_num = len(env.action_space)
            tracker_id = env.unwrapped.tracker_id
            target_id = env.unwrapped.target_id
            tracker = PoseTracker(env.action_space[0]) # TODO support multi trackers
            tracker_random = RandomAgent(env.action_space[0])
            count_step = 0
            t0 = time.time()
            agents_num = len(obs)
            C_rewards = np.zeros(agents_num)
            print('eps:', eps, 'agents_num:', agents_num)
            image = []
            act = []
            reward = []
            direction=[]
            distance=[]
            flag = 0
            pos = []
            consecutive_low_reward_steps = 0  #
            k = 10  #
            x = 0.1  #
            while True:
                st_time =time.time()
                obj_poses = env.unwrapped.obj_poses
                action_auto = tracker.act(obj_poses[tracker_id], obj_poses[target_id])
                actions = [tracker.act(obj_poses[tracker_id], obj_poses[target_id])]
                flag -= 1
                if random.random() < 0.3 or flag > 0:
                    actions[0] = tracker_random.act(obj_poses)
                    if flag <= 0:
                        action_tmp = actions[0]
                        flag = random.randint(3, 6)
                    else:
                        actions[0] = action_tmp
                    # print('pid+noise:', actions)


                act.append(action_auto)
                obs, rewards, done, info= env.step(actions)
                print(count_step,":",rewards)
                C_rewards += rewards
                count_step += 1
                cv2.imshow('obs', obs[0])
                cv2.waitKey(1)
                image.append(obs[0])
                # pos.append(np.array(info['Pose']))
                reward.append(rewards)

                #
                if rewards < x:
                    consecutive_low_reward_steps += 1
                else:
                    consecutive_low_reward_steps = 0

                #
                if consecutive_low_reward_steps >= k:
                    print(f"Paused eps {eps} due to low rewards for {k} consecutive steps.")
                    # cmd = f'vset /action/game/pause'
                    # env.unwrapped.unrealcv.client.request(cmd)
                    # print("Press 'y' to start controlling the agent")
                    consecutive_high_reward_steps = 0
                    while True:
                        # cmd = f'vset /action/game/resume'
                        # env.unwrapped.unrealcv.client.request(cmd)
                        action = get_key_action()
                        # print('keyboard action:', action)
                        action = list(action[0])

                        obs, rewards, done, info = env.step([action])
                        cv2.imshow('obs', obs[0])
                        cv2.waitKey(1)
                        count_step += 1
                        act.append(action)
                        reward.append(rewards)
                        image.append(obs[0])


                        if rewards > 0.7:
                            consecutive_high_reward_steps += 1
                        else:
                            consecutive_high_reward_steps = 0


                        if consecutive_high_reward_steps >= 10:
                            print("Resumed normal operation due to high rewards for 10 consecutive steps.")
                            break

                if args.render:
                    cv2.imshow('show', obs[0])
                    cv2.waitKey(1)
                if done:
                    fps = count_step / (time.time() - t0)
                    Total_rewards += C_rewards[0]
                    is_first = np.array([True] + (len(image) - 1) * [False])
                    is_last = np.array((len(image) - 1) * [False] + [True])
                    dict = {
                        'action': act,
                        'image': image,
                        'reward': reward,
                        # 'is_first': is_first,
                        # 'is_last': is_last,
                        # 'dis': distance,
                        # 'state': pos ,
                        # 'direction':direction
                    }

                    save_dir = os.path.join(
                        'D:\\pid_data\\',
                        'PID+noise_V1_240px_GT_2st_' + "%04d" % int(eps) + "-%03d" % count_step + '.pt')
                    torch.save(dict,save_dir)
                    break

        print('Finished')
        env.close()
    except KeyboardInterrupt:
        print('exiting')
        env.close()