import sys

import numpy as np
from gym_unrealcv.envs.unrealcv_EQA_general import UnrealCvEQA_general
import cv2
from scipy.spatial.transform import Rotation as R
from PIL import ImageDraw,ImageFont
import transforms3d
import unrealcv
import json
def load_json_file(file_path):
    """
    Load a JSON file and return its content.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        dict: Content of the JSON file.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


class UnrealCvEQA_DATA(UnrealCvEQA_general):
    def __init__(self,
                 setting_file,  # the setting file to define the task
                 task_file=None,  # the file to define the task
                 action_type='Discrete',  # 'discrete', 'continuous'
                 observation_type='Color',  # 'color', 'depth', 'rgbd', 'Gray'
                 resolution=(160, 160),
                 reward_type = 'distance',
                 reset_type=0,
                 docker=False
                 ):
        super(UnrealCvEQA_DATA, self).__init__(setting_file=setting_file,  # the setting file to define the task
                                    action_type=action_type,  # 'discrete', 'continuous'
                                    observation_type=observation_type,  # 'color', 'depth', 'rgbd', 'Gray'
                                    resolution=resolution,
                                    reset_type=reset_type)
        


    
    def step(self,action):
        obs, rewards, termination, truncation,info = super(UnrealCvEQA_DATA, self).step(action)
        
        return obs, rewards, termination, truncation,info
    
    def reset(self,seed=None, options=None):
        obs, info = super(UnrealCvEQA_DATA, self).reset(seed=seed,options=options)
        return obs, info