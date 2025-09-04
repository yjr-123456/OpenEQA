import sys

import numpy as np
from gym_unrealcv.envs.base_env import UnrealCv_base
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


class UnrealCvObjectInfoCollection(UnrealCv_base):
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
        super(UnrealCvObjectInfoCollection, self).__init__(setting_file=setting_file,  # the setting file to define the task
                                    action_type=action_type,  # 'discrete', 'continuous'
                                    observation_type=observation_type,  # 'color', 'depth', 'rgbd', 'Gray'
                                    resolution=resolution,
                                    reset_type=reset_type)
        


    
    def step(self,action):
        obs, rewards, termination, truncation,info = super(UnrealCvObjectInfoCollection, self).step(action)

        return obs, rewards, termination, truncation,info
    
    def reset(self,seed=None, options=None):
        obs, info = super(UnrealCvObjectInfoCollection, self).reset(seed=seed,options=options)

        info['obj_info'] = self.get_obj_info()
        return obs, info
    
    def get_obj_info(self):
        object_name = self.unrealcv.get_objects()
        return self.unrealcv.build_pose_dic(object_name)
    