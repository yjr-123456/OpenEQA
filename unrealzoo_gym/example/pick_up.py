import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import argparse
#import gym_unrealcv
import gym
from gym import wrappers
import cv2
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, agents, augmentation, configUE, sample_agent_configs
import json
from pynput import keyboard
from datetime import datetime
import time
import numpy as np


if __name__ == '__main__':
    # env name
    env_name = "Map_ChemicalPlant_1"
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env_id", nargs='?', default=f'UnrealCvEQA_general-{env_name}-DiscreteColorMask-v0',
                        help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=0, help='random seed')
    parser.add_argument("-t", '--time-dilation', dest='time_dilation', default=-1, help='time_dilation to keep fps in simulator')
    parser.add_argument("-n", '--nav-agent', dest='nav_agent', action='store_true', help='use nav agent to control the agents')
    parser.add_argument("-d", '--early-done', dest='early_done', default=-1, help='early_done when lost in n steps')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')
    parser.add_argument("-c", '--ifonly-counting', dest='if_cnt', action='store_true', help='if only counting the number of agents in the scene')
    parser.add_argument("-g", "--reachable-points-graph", dest='graph_path', default=f"./agent_configs_sampler/points_graph/{env_name}/environment_graph.gpickle", help='use reachable points graph')
    parser.add_argument("-w", "--work-dir", dest='work_dir', default="E:/EQA/unrealzoo_gym/example", help='work directory to save the data')
    args = parser.parse_args()
    
    env = gym.make(args.env_id)
    
    # some configs
    currpath = os.path.dirname(os.path.abspath(__file__))
    agents_category = ['player']
    animation = 'pick_up'  # 'crouch', 'liedown', 'pick_up', 'in_vehicle'
    obs_name = "BP_Character_C_1"
    action = [-1]
    # work_dir = "/home/yjr/UnrealZoo/gym_unrealzoo-E034/example/random"
    # env wrappers
    if int(args.time_dilation) > 0:  # -1 means no time_dilation
        env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
    if int(args.early_done) > 0:  # -1 means no early_done
        env = early_done.EarlyDoneWrapper(env, int(args.early_done))
    if args.monitor:
        env = monitor.DisplayWrapper(env)
    
    # agent sample pre-define
    agent_categories = ['player']      
    env = configUE.ConfigUEWrapper(env, offscreen=False)
    save_cnt = 0
    try:
        env.reset()
        unwrapped_env = env.unwrapped
        # set location
        unwrapped_env.unrealcv.set_obj_location(obs_name, [3986, -1954, -12707])
        unwrapped_env.unrealcv.set_obj_rotation(obs_name, [0, 26.519, 0.000])
        # set animation
        # pick_up_class = "BP_GrabMoveDrop_C"
        # unwrapped_env.unrealcv.new_obj(pick_up_class,f"{pick_up_class}_1", [415, 355,45], [0,0,0])
        # unwrapped_env.unrealcv.set_obj_scale(f"{pick_up_class}_1", [0.2, 0.2, 0.2])
        time.sleep(3)
        unwrapped_env.unrealcv.set_animation(obs_name, animation)
        time.sleep(3)
        env.step([-2])
        unwrapped_env.unrealcv.set_animation(obs_name, animation)

    except KeyboardInterrupt:
        env.close()
    finally:
        if args.render:
            cv2.destroyAllWindows()




