from gym_unrealcv.envs.base_env import UnrealCv_base
import numpy as np
import random
import time
import math
import transforms3d
'''
Tasks: The task is to make the agents find the target(injured person) and rescue him. 
The agents are allowed to communicate with others.
The agents observe the environment with their own camera.
The agents are rewarded based on the distance and orientation to the target.
The episode ends when the agents reach the target(injured agent) or the maximum steps are reached.
'''

class Nav_Data(UnrealCv_base):
    def __init__(self,
                 setting_file,  # the setting file to define the task
                 action_type='Discrete',  # 'discrete', 'continuous'
                 observation_type='Color',  # 'color', 'depth', 'rgbd', 'Gray'
                 reward_type='distance',  # 'sparse', 'shared', 'individual'
                 resolution=(160, 160),
                 reset_type=0,
                 docker = False,
                 nav="Random"
                 ):
        super(Nav_Data, self).__init__(setting_file=setting_file,  # the setting file to define the task
                                         action_type=action_type,  # 'discrete', 'continuous'
                                         observation_type=observation_type,  # 'color', 'depth', 'rgbd', 'Gray'
                                         reward_type=reward_type,  # 'sparse', 'shared', 'individual'
                                         resolution=resolution,
                                         reset_type=reset_type,
                                         docker=docker,
                                         nav=nav
                                         )

    def step(self, action):
        obs, rewards, done, info = super(Nav_Data, self).step(action)
        
        return obs, rewards, done, info

    def reset(self):
        # initialize the environment
        states, info = super(Nav_Data, self).reset()
        print(self.nav_configs)
        self.target_name = self.nav_configs["Target_name"]
        self.start_pose = self.nav_configs["Start_pose"]
        self.instruction = self.nav_configs["Instruction"]
        self.info = {
            'Target_name': self.target_name,
            'Start_pose': self.start_pose,
            'Instruction': self.instruction
        }
        # set navigator location
        if self.start_pose is not None:
            self.unrealcv.set_obj_location(self.player_list[self.protagonist_id], self.start_pose[:3])
            self.unrealcv.set_obj_rotation(self.player_list[self.protagonist_id], self.start_pose[3:])
        # get target location
        target_location = self.unrealcv.get_obj_location(self.target_name)
        target_rotation = self.unrealcv.get_obj_rotation(self.target_name)
        self.info['Target_pose'] = target_location + target_rotation

        # set viewpoint
        self.set_topview(self.start_pose[:3], self.cam_id[0])
        time.sleep(0.5)
        # get obs
        obs, self.obj_poses, self.img_show = self.update_observation(self.player_list, self.cam_list, self.cam_flag, self.observation_type)
        # calculate relative pose
        self.info['Relative_pose'] = self.caculate_relative_pose(self.start_pose, self.info['Target_pose'])
        return obs,self.info

    def caculate_relative_pose(self, cur_obj_pose,tar_obj_pose):
        direction_vector = np.array(tar_obj_pose[:3]) - np.array(cur_obj_pose[:3])
        pitch,yaw,roll = cur_obj_pose[3:]
        pitch = np.radians(pitch)
        roll = np.radians(roll)
        
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
        roll = np.radians(roll)
        rot = transforms3d.euler.euler2mat(-roll, -pitch, yaw, 'sxyz')
        rot_wl = np.linalg.inv(rot)
        # Transform the direction vector to obj1's frame
        local_direction = np.dot(rot_wl, direction_vector)
                    
        # Calculate the horizontal angle in the XY plane
        angle = math.degrees(math.atan2(local_direction[1], local_direction[0]))
                    
        # Determine the relative direction
        direction_desc = self._get_direction_description(angle)
        # direction_desc = self._get_direction_description(local_direction)
        return direction_desc
    
    def _get_direction_description(self, angle):
        """Convert an angle to a direction description"""
        # Normalize angle to -180 to 180
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        
        # Define direction based on angle
        if -22.5 <= angle <= 22.5:
            return "in front of"
        elif 22.5 < angle <= 67.5:
            return "to the front-right of"
        elif 67.5 < angle <= 112.5:
            return "to the right of"
        elif 112.5 < angle <= 157.5:
            return "to the back-right of"
        elif angle > 157.5 or angle < -157.5:
            return "behind"
        elif -157.5 <= angle < -112.5:
            return "to the back-left of"
        elif -112.5 <= angle < -67.5:
            return "to the left of"
        elif -67.5 <= angle < -22.5:
            return "to the front-left of"
        else:
            return "near" 