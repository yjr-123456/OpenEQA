import sys
import os
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)  # Add parent directory to sys.path
import argparse
#import gym_unrealcv
import gymnasium as gym
# from gymnasium import wrappers
import cv2
import time
import numpy as np
import os
# import torch
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, agents, augmentation, configUE
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import tqdm
# class RandomAgent(object):
#     """The world's simplest agent!"""
#     def __init__(self, action_space):
#         self.action_space = action_space
#         self.count_steps = 0
#         self.action = self.action_space.sample()

#     def act(self, observation, keep_steps=10):
#         self.count_steps += 1
#         if self.count_steps > keep_steps:
#             self.action = self.action_space.sample()
#             self.count_steps = 0
#         else:
#             return self.action
#         return self.action

#     def reset(self):
#         self.action = self.action_space.sample()
#         self.count_steps = 0

os.environ["UnrealEnv"] = "/Volumes/KINGSTON/UnrealEnv"
class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space
        self.count_steps = 0
        
        # list type action_space
        if isinstance(self.action_space, list):
            # if list, sample actions for every agent
            self.action = [space.sample() if hasattr(space, 'sample') 
                          else random.randint(0, 7) for space in self.action_space]
        else:
            # common gym space obj
            self.action = self.action_space.sample()

    def act(self, keep_steps=10):
        self.count_steps += 1
        if self.count_steps > keep_steps:
            # sample new actions
            if isinstance(self.action_space, list):
                self.action = [space.sample() if hasattr(space, 'sample') 
                              else random.randint(0, 7) for space in self.action_space]
            else:
                self.action = self.action_space.sample()
            self.count_steps = 0
        
        return self.action

    def reset(self):
        # same logic
        if isinstance(self.action_space, list):
            self.action = [space.sample() if hasattr(space, 'sample') 
                          else random.randint(0, 7) for space in self.action_space]
        else:
            self.action = self.action_space.sample()
        self.count_steps = 0


import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time


if __name__ == '__main__':
    # 定义要采集的环境列表

    
    # 遍历每个环境进行采集
    try:
        parser = argparse.ArgumentParser(description=None)
        parser.add_argument("-e", "--env_id", nargs='?', default=f'UnrealCv_base-ModularSciFiVillage-DiscreteColor-v0',
                            help='Select the environment to run')
        parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
        parser.add_argument("-s", '--seed', dest='seed', default=42, help='random seed')
        parser.add_argument("-t", '--time-dilation', dest='time_dilation', default=-1, help='time_dilation to keep fps in simulator')
        parser.add_argument("-n", '--nav-agent', dest='nav_agent', action='store_true', help='use nav agent to control the agents')
        parser.add_argument("-d", '--early-done', dest='early_done', default=-1, help='early_done when lost in n steps')
        parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')
        parser.add_argument("--ue_log_dir", default=os.path.join(current_dir, "unreal_log_path"), help="unreal engine logging directory")

        args = parser.parse_args([])  # 使用空列表避免从命令行解析参数
        env = gym.make(args.env_id)
        if int(args.time_dilation) > 0:
            env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
        if int(args.early_done) > 0:
            env = early_done.EarlyDoneWrapper(env, int(args.early_done))
        if args.monitor:
            env = monitor.DisplayWrapper(env)
        # agent_num = len(env.unwrapped.safe_start)
        env.unwrapped.safe_start = [     [
                    -1368.163,
                    -107.027,
                    1038.211,
                    0,
                    112.74871326841138,
                    0
                ]]
        env.unwrapped.refer_agents_category = ['player']
        env.unwrapped.ue_log_path = args.ue_log_dir
        obj_name = "BP_Character_C_1"
        target_config = {
            "player": {
                "name": [
                    "player_1_IKHBA6",
                    "player_2_IKHBA6",
                    "player_3_IKHBA6",
                    "player_4_IKHBA6",
                ],
                "app_id": [
                    1,2,3,4
                ],
                "animation": [
                    "pick_up",
                    "stand",
                    "liedown",
                    "crouch"
                ],
                "start_pos": [
                                                 [
                    -1710.718,
                    -91.984,
                    838.275,
                    0.0,
                    101.881,
                    0.0
                ],
                [
                    -1623.088,
                    -282.06,
                    838.033,
                    0,
                    126.1051676449647,
                    0
                ],
                [
                    -1607.882,
                    -534.925,
                    838.275,
                    0,
                    178.59330765368298,
                    0
                ],
                [
                    -1511.918,
                    -15.845,
                    838.211,
                    0,
                    157.67281191663312,
                    0
                ]
                ],
                "type": [
                    "None",
                    "None",
                    "None",
                    "None"
                ]
            }
        }
        
        # for app_th in range(4):
            # appid_list = [appid+app_th*5 for appid in range(0, 5)]
        # target_config['player']['app_id'] = appid_list
        env.unwrapped.target_configs = target_config
        env = augmentation.RandomPopulationWrapper(env, num_min=5, num_max=5, height_bias = 100)
        env.reset(seed=int(args.seed))
        time.sleep(10)
        # for app_id in range(0,20):
        #     player_id = app_id % 5 + 1
        #     obj_name = f"player_{player_id}_IKHBA6"
        #     print(f"=========testing {obj_name} appearance=========")
        #     for i in range(1,app_id + 1):
        #         time.sleep(3.0)
        #         env.unwrapped.unrealcv.set_appearance(obj_name, app_id+1)
        #         print(f"=========set {obj_name} appearance to {app_id+1}=========")
        env.close()
    
    except KeyboardInterrupt:
        print(f'用户中断')
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        env.close()  
    finally:
        env.close()
