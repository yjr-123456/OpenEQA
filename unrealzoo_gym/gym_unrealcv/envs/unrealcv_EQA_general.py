import numpy as np
from gym_unrealcv.envs.base_env import UnrealCv_base
import time
# from gym_unrealcv.envs.utils import misc, reward



class UnrealCvEQA_general(UnrealCv_base):
    def __init__(self,
                 setting_file,  # the setting file to define the task
                 task_file=None,  # the file to define the task
                 action_type='Discrete',  # 'discrete', 'continuous'
                 observation_type='Color',  # 'color', 'depth', 'rgbd', 'Gray'
                 resolution=(1080, 1080),
                 reward_type = 'distance',
                 reset_type=0,
                 docker=False
                 ):
        super(UnrealCvEQA_general, self).__init__(setting_file=setting_file,  # the setting file to define the task
                                    action_type=action_type,  # 'discrete', 'continuous'
                                    observation_type=observation_type,  # 'color', 'depth', 'rgbd', 'Gray'
                                    resolution=resolution,
                                    reset_type=reset_type)


        # self.cam_id = self.setting['cam_id']
        #self.target_list =self.env_configs['targets']['Point']
        
        self.player = self.player_list
        self.resolution = resolution
        self.observation_type = observation_type
        # assert self.observation_type == 'Color' or self.observation_type == 'Depth' or self.observation_type == 'Rgbd' or self.observation_type == 'Mask'
        # self.observation_space = self.unrealcv.define_observation(self.cam_id, self.observation_type, 'direct')

        # define reward type
        # distance, bbox, bbox_distance,
        # self.reward_type = reward_type
        self.reward_type = 'distance'
        # self.reward_function = reward.Reward()
        self.trigger_count = 0
        

    def step(self, action):
        obs, rewards, termination, truncation,info = super(UnrealCvEQA_general, self).step(action)

        #detect if the agent collision with environment
        if self.unrealcv.get_hit(self.player[self.protagonist_id]) == 0:
            info['Collision'] = 0
        else:
            info['Collision'] += 1
        # info['Pose'] = self.unrealcv.get_pose(self.cam_id, 'soft') #for virtual camera
        info['Pose'] = self.unrealcv.get_obj_pose(self.player[self.protagonist_id])
        # calculate relative pose
        # print(self.target_list)
        # pose_obs, relative_pose_2d = self.unrealcv.get_pose_states([info['Pose'], self.targets_pos[self.target_list[0]]])
        # info['relative_pose'] = np.array([relative_pose_2d[0][1][0], relative_pose_2d[0][1][1],
        #                                   self.targets_pos[self.target_list[0]][2] - info['Pose'][
        #                                       2]])  # distance,direction,height : point to player
        

        #reward computation
        # distance, self.target_id = self.select_target_by_distance(info['Pose'][:3], self.targets_pos)
        # info['Target'] = self.targets_pos[self.target_id]
        # info['Direction'] = misc.get_direction(info['Pose'], self.targets_pos[self.target_id])

        # # calculate reward according to the distance to target object
        # if 'distance' in self.reward_type:
        #     # info['Reward'] = self.reward_function.reward_distance(distance)
        #     relative_oir_norm = np.fabs(info['Direction']) / 90.0
        #     reward_norm = np.tanh(self.reward_function.reward_distance(distance) - relative_oir_norm)
        #     info['Reward'] = reward_norm
        # else:
        #     info['Reward'] = 0

        # # if collision detected, the episode is done and reward is -1
        # if info['Collision'] > 10 or info['Pose'][2] < self.height/2:
        #     info['Reward'] = -1
        #     info['Done'] = True

        # if distance < 300 and np.fabs(info['Direction']) < 10:
        #     info['Success'] = True
        #     info['Done'] = True
        #     info['Reward'] = 100

        
        # save the trajectory
        self.trajectory.append(info['Pose'][:6])    #loaction and rotation
        info['Trajectory'] = self.trajectory        
        #self.count_steps += 1
        if self.count_steps > self.max_steps:
            info['truncation'] = True
        #print("pose: ", info['Pose'])
        return obs, info['Reward'], info['termination'],info['truncation'], info

    def reset(self, ):
        # double check the resetpoint, it is necessary for random reset type
        observations,info = super(UnrealCvEQA_general, self).reset()

        current_pose = self.unrealcv.get_pose(self.cam_id[self.protagonist_id])
        #print("current_pose: ", current_pose)
        #self.targets_pos = self.unrealcv.build_pose_dic(self.target_list)

        #self.unrealcv.set_obj_color(self.target_list[0], (255, 255, 255))
        #state = self.unrealcv.get_observation(self.cam_id, self.observation_type)
        #observations, self.obj_poses, self.img_show = self.update_observation(self.player_list, self.cam_list, self.cam_flag, self.observation_type)

        self.trajectory = []
        self.trajectory.append(current_pose)
        self.trigger_count = 0
        self.count_steps = 0
        #self.reward_function.dis2target_initial, self.targetID_last = \
        #    self.select_target_by_distance(current_pose, self.targets_pos)
        self.info["obs"] = observations
        self.info["Third_View_Pose"] = current_pose
        self.info["Pose"] = self.obj_poses[0]
        print("agent pose",self.info["Pose"])
        #print(observations.shape)
        time.sleep(0.5)
        return observations, self.info

    def seed(self, seed=None):
        return seed

    def render(self, mode='rgb_array', close=False):
        if close:
            self.unreal.close()
        return self.unrealcv.img_color

    def close(self):
        self.unrealcv.client.disconnect()
        self.ue_binary.close()

    def get_action_size(self):
        return len(self.action)

    def select_target_by_distance(self, current_pos, targets_pos):
        # find the nearest target, return distance and targetid
        target_id = list(self.targets_pos.keys())[0]
        # distance_min = self.unrealcv.get_distance(targets_pos[target_id], current_pos, 2)
        distance_min = self.unrealcv.get_distance(targets_pos[target_id], current_pos, 3)

        for key, target_pos in targets_pos.items():
            # distance = self.unrealcv.get_distance(target_pos, current_pos, 2)
            distance = self.unrealcv.get_distance(target_pos, current_pos, 3)
            if distance < distance_min:
                target_id = key
                distance_min = distance
        return distance_min, target_id
    

        