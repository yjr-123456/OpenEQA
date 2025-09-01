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

def calculate_distance(pos1, pos2):
    """计算两个位置之间的欧几里得距离"""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)


if __name__ == '__main__':
    # env name
    env_name = "SuburbNeighborhood_Day"
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env_id", nargs='?', default=f'UnrealCv_base-{env_name}-DiscreteColorMask-v0',
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
   
    # env wrappers
    if int(args.time_dilation) > 0:  # -1 means no time_dilation
        env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
    if int(args.early_done) > 0:  # -1 means no early_done
        env = early_done.EarlyDoneWrapper(env, int(args.early_done))
    if args.monitor:
        env = monitor.DisplayWrapper(env)
    
    # agent sample pre-define
    env = augmentation.RandomPopulationWrapper(env, num_min=3, num_max=3, agent_category=agents_category)
    agent_categories = ['player']      
    env = configUE.ConfigUEWrapper(env, offscreen=False)
    save_cnt = 0
    env.reset()
    loc1 = [-533,-1413, 98]
    loc2 = [-749,-915, 98]
    loc3 = [-927,-149, 98]
    bias = [50,50,0]
    loc1 = [loc1[0]+bias[0], loc1[1]+bias[1], loc1[2]+bias[2]]
    loc2 = [loc2[0]+bias[0], loc2[1]+bias[1], loc2[2]+bias[2]]
    time.sleep(10)
    try:
        path = env.unwrapped.unrealcv.nav_to_goal_bypath("BP_Character_C_1", loc1)
        print(path)
        time.sleep(300)
        # env.unwrapped.unrealcv.set_obj_location("BP_Character_C_1", loc3)
        # path = env.unwrapped.unrealcv.nav_to_goal_bypath("BP_Character_C_2", loc2)
        # print(path)

    except KeyboardInterrupt:
        env.close()
    finally:
        env.close()
        if args.render:
            cv2.destroyAllWindows()




