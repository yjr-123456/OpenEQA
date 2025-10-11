import warnings
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gym_unrealcv.envs.utils import misc
from unrealcv.launcher import RunUnreal
from gym_unrealcv.envs.agent.character import Character_API
import random
import sys
import time
import math
import socket
''' 
It is a base env for general purpose agent-env interaction, including single/multi-agent navigation, tracking, etc.
Observation : raw color image and depth
Action:  Discrete/Continuous
Done : define by the task wrapper
'''

def calculate_distance(pos1, pos2):
    """计算两个位置之间的欧几里得距离"""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)

def send_pid(pid, host='127.0.0.1', port=50007):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(str(pid).encode())

# TODO: agent apis for blueprints commands
# TODO: config env by parapmeters
# TODO: maintain a general agent list
class UnrealCv_base(gym.Env):
    """
    A base environment for general purpose agent-environment interaction, including single/multi-agent navigation, tracking, etc.
    Observation: color image, depth image, rgbd image, mask image, pose
    Action: Discrete, Continuous, Mixed
    Done: defined by the task wrapper
    """
    def __init__(self,
                 setting_file,  # the setting file to define the task
                 action_type='Discrete',  # 'discrete', 'continuous'
                 observation_type='Color',  # 'color', 'depth', 'rgbd', 'Gray'
                 reward_type='distance',  # 'sparse', 'dense'
                 resolution=(1080, 1080),
                 reset_type=0,
                 docker=False
                 ):
        """
        Initialize the UnrealCv_base environment.

        Args:
            setting_file (str): The setting file to define the task and environments (path2binary, action space, reset area).
            action_type (str): Type of action space ('Discrete', 'Continuous').
            observation_type (str): Type of observation space ('Color', 'Depth', 'Rgbd', 'Gray').
            resolution (tuple): Resolution of the observation space.
            reset_type (int): Type of reset.
        """
        setting = misc.load_env_setting(setting_file)
        self.env_name = setting['env_name']
        self.max_steps = setting['max_steps'] if 'max_steps' in setting else 100
        self.height = setting['height']
        self.cam_id = [setting['third_cam']['cam_id']]
        self.agent_configs = setting['agents']
        self.target_configs = setting['target_configs'] if 'target_configs' in setting else None

        self.env_configs = setting["env"]
        # TODO
        self.vehicle_zones = setting["vehicle_zones"] if "vehicle_zones" in setting else []
        self.height_bias = setting["height_bias"] if "height_bias" in setting else {
            "car": 0,
            "animal": [0, 300, 150],
            "drone": 200
        }

        self.agents = misc.convert_dict(self.agent_configs)
        self.refer_agents = misc.convert_dict(self.agent_configs)
        self.target_agents = misc.convert_dict(self.target_configs) if self.target_configs is not None else {}
        self.reset_type = reset_type
        self.target_list = []
        self.is_eval = False
        # self.target_camera_list = []

        # TODO: it is useless.
        self.character = {
            'player': [],  # the list of player to control
            'npc': [],  # the list of Non-player character
            'freeze': [],  # the list of player that exists in the scene, but it is frozen
        }

        self.height_top_view = setting['third_cam']['height_top_view']

        # self.env_obj_list = self.env_configs[""]
        self.objects_list = []
        self.reset_area = setting['reset_area']

        self.safe_start = setting['safe_start']
        self.interval = setting['interval']
        self.random_init = setting['random_init']
        self.start_area = self.get_start_area(self.safe_start[0], 500) # the start area of the agent, where we don't put obstacles
        # self.max_steps=setting['max_steps']
        self.count_eps = 0
        self.count_steps = 0
        self.target_start = []
        self.pickup_list = []
        # env configs
        self.docker = docker
        self.resolution = resolution
        self.display = None
        self.use_opengl = False
        self.offscreen_rendering = False
        self.nullrhi = False
        self.gpu_id = None  # None means using the default gpu
        self.sleep_time = 5
        self.launched = False
        self.comm_mode = 'tcp'

        self.agents_category = ['player'] # the agent category we use in the env
        self.refer_agents_category = ['player']  # the agent category we use as a reference to add new agents
        self.protagonist_id = 0

        # init agents
        self.player_list = list(self.agents.keys())
        self.cam_list = [self.agents[player]['cam_id'] for player in self.player_list]
        


        # define action space
        self.action_type = action_type
        assert self.action_type in ['Discrete', 'Continuous', 'Mixed']
        self.action_space = spaces.Tuple([self.define_action_space(self.action_type, self.agents[obj]) for obj in self.player_list])

        # define observation space,
        # color, depth, rgbd,...
        self.observation_type = observation_type
        assert self.observation_type in ['Color', 'Depth', 'Rgbd', 'Gray', 'CG', 'Mask', 'Pose','MaskDepth','ColorMask']
        self.observation_space = spaces.Tuple([self.define_observation_space(self.cam_list[i], self.observation_type, resolution)
                                  for i in range(len(self.player_list))])

        # config unreal env
        if 'linux' in sys.platform:
            env_bin = setting['env_bin']
        elif 'darwin' in sys.platform:
            env_bin = setting['env_bin_mac']
        elif 'win' in sys.platform:
            env_bin = setting['env_bin_win']
        if 'env_map' in setting.keys():
            env_map = setting['env_map']
        else:
            env_map = None

        self.ue_binary = RunUnreal(ENV_BIN=env_bin, ENV_MAP=env_map)

        self.info = dict(
            Collision=0,
            termination=False,
            truncation=False,
            Reward=0.0,
            Action=None,
            Pose=[],
            Steps=0,
            Direction=None,
            Distance=None,
            Color=None,
            Depth=None,
            Relative_Pose=[],
            Success=False
        )
        self.ue_log_path = None
        self.send_pid = False
        self.watchdog_port = 50007



    def step(self, actions):
        """
        Execute one step in the environment.

        Args:
            actions (list): List of actions to be performed by the agents.

        Returns:
            tuple: Observations, rewards, done flag, and additional info.
        """
        self.info['Action'] = actions
        actions2move, actions2turn, actions2animate = self.action_mapping(actions, self.player_list)
        move_cmds = [self.unrealcv.set_move_bp(obj, actions2move[i], return_cmd=True) for i, obj in enumerate(self.player_list) if actions2move[i] is not None]
        head_cmds = [self.unrealcv.set_cam(obj, self.agents[obj]['relative_location'], actions2turn[i], return_cmd=True) for i, obj in enumerate(self.player_list) if actions2turn[i] is not None]
        anim_cmds = [self.unrealcv.set_animation(obj, actions2animate[i], return_cmd=True) for i, obj in enumerate(self.player_list) if actions2animate[i] is not None]
        self.unrealcv.batch_cmd(move_cmds+head_cmds+anim_cmds, None)
        self.count_steps += 1
        self.info['Steps'] = self.count_steps

        time.sleep(1)
        
        # get states self.agents
        obj_poses, cam_poses, imgs, masks, depths = self.unrealcv.get_pose_img_batch(self.player_list, self.cam_list, self.cam_flag)
        self.obj_poses = obj_poses
        observations = self.prepare_observation(self.observation_type, imgs, masks, depths, obj_poses)
        self.img_show = self.prepare_img2show(self.protagonist_id, observations)

        pose_obs, relative_pose = self.get_pose_states(obj_poses)

        # prepare the info
        self.info['Pose'] = obj_poses
        self.info['Relative_Pose'] = relative_pose
        self.info['Pose_Obs'] = pose_obs
        self.info['Reward'] = np.zeros(len(self.player_list))

        #get camera info
        # self.info['cam_info'] = self.unrealcv.get_cam_info(self.agents[self.player_list[self.protagonist_id]]['cam_id'])
        #print(self.info['cam_info'])
        return observations, self.info['Reward'], self.info['termination'], self.info['truncation'],self.info

    def reset(self,seed=None, options=None):
        """
        Reset the environment to its initial state.

        Returns:
            np.array: Initial observations.
        """
        super().reset(seed=seed)
        if not self.launched:  # first time to launch
            self.launched = self.launch_ue_env(self.ue_log_path)
            self.init_agents()
            self.init_objects()

        self.count_close = 0
        self.count_steps = 0
        self.count_eps += 1
        self.info['ue_pid'] = self.get_ue_pid()
        # stop move and disable physics
        for i, obj in enumerate(self.player_list):
            if self.agents[obj]['agent_type'] in self.agents_category:
                if not self.agents[obj]['internal_nav']:
                    # self.unrealcv.set_move_bp(obj, [0, 100])
                    # self.unrealcv.set_max_speed(obj, 100)
                    continue
                    # self.unrealcv.set_phy(obj, 1)
            elif self.agents[obj]['agent_type'] == 'drone':
                self.unrealcv.set_move_bp(obj, [0, 0, 0, 0])
                time.sleep(0.5)
                self.unrealcv.set_phy(obj, 1)

        # reset target location
        # init_poses = self.sample_init_pose(self.random_init, len(self.player_list))
        init_poses = self.safe_start + self.target_start
        print("len init_poses:", len(init_poses),"len player_list:", len(self.player_list)) 
        print("init_poses:\n", init_poses)
        print("player_list:\n", self.player_list)
        # assert len(init_poses) == len(self.player_list), "The number of initial poses must match the number of players."
        for i, obj in enumerate(self.player_list):
            self.unrealcv.set_obj_location(obj, init_poses[i][0:3])

            # for name in self.player_list:
            #     print("num of agent:",len(self.player_list)-1)
            #     print("location of obj:", name, "is", self.unrealcv.get_obj_location(name))
            
            self.unrealcv.set_obj_rotation(obj, init_poses[i][-3:])
            if self.agents[obj]['agent_type'] in ['animal','car','drone','player']:
                time.sleep(0.5)
                self.unrealcv.set_phy(obj, 1)
                time.sleep(0.8)
                self.unrealcv.set_phy(obj, 0)
            elif self.agents[obj]['agent_type'] == 'motorbike':
                time.sleep(0.5)
                self.unrealcv.set_phy(obj, 0)
            self.unrealcv.set_cam(obj, self.agents[obj]['relative_location'], self.agents[obj]['relative_rotation'])
        # set top view
        self.set_topview(init_poses[self.protagonist_id], self.cam_id[0])
        pick_cnt = 0

        if len(self.target_list) != 0:
            # check agent
            self.target_list = self.check_agent()
            print("target list after check:\n",self.target_list)
            # for i, obj in enumerate(self.target_list):
            #     app_id = self.target_agents[obj]['app_id']
            #     if self.agents[obj]['agent_type'] not in ['car','motorbike']:
            #         self.unrealcv.set_appearance(obj, app_id)
            #         time.sleep(3.0)
            #         print(f"======set appearance for {obj}, app_id:{app_id}")
            for i, obj in enumerate(self.target_list):
                #self.unrealcv.set_obj_color(obj, self.agent_color[self.agents[obj]['color']])
                animation = self.target_agents[obj]['animation']
                if animation == 'None' or animation == 'stand':
                    continue
                elif animation == 'crouch' or animation == 'liedown':
                    self.unrealcv.set_animation(obj, animation)
                elif animation == 'pick_up':
                    batch_id = obj.split('_')[-1]  # get the batch id from the object name
                    loca = self.unrealcv.get_obj_location(obj)
                    rot = self.unrealcv.get_obj_rotation(obj)
                    theta = np.deg2rad(rot[1])
                    bias = [50*np.cos(theta-np.pi/2), 50*np.sin(theta-np.pi/2), 0]
                    loc = [loca[i] + bias[i] for i in range(3)]
                    pick_up_class = "BP_GrabMoveDrop_C"
                    pick_up_name = f"{pick_up_class}_{batch_id}_{pick_cnt}"
                    self.pickup_list.append(pick_up_name)
                    self.unrealcv.new_obj(pick_up_class,pick_up_name,loc, rot)
                    self.unrealcv.set_obj_color(pick_up_name, np.random.randint(0, 255, 3))
                    time.sleep(2.0)
                    self.unrealcv.set_animation(obj, animation)
                    time.sleep(2.0)
                    cur_loc = self.unrealcv.get_obj_location(obj)
                    cur_rot = self.unrealcv.get_obj_rotation(obj)
                    agent_type = list(obj.split('_'))[0]  # get the agent type from the object name
                    index = self.target_configs[agent_type]['name'].index(obj)
                    self.target_configs[agent_type]['start_pos'][index] = cur_loc + cur_rot
                    pick_cnt += 1
                elif animation == 'in_vehicle':
                    # search for the vehicle
                    has_vehicle = any(self.agents[obj]['agent_type'] in ['car', 'motorbike'] for obj in self.target_list)
                    if has_vehicle:
                        # 如果有车辆类型，获取所有车辆对象
                        vehicle_list = [obj for obj in self.player_list 
                                    if self.agents[obj]['agent_type'] in ['car', 'motorbike']]
                    else:
                        vehicle_list = []
                    
                    if len(vehicle_list) > 0:
                        obj_loc = self.unrealcv.get_obj_location(obj)
                        min_distance = float('inf')
                        nearest_vehicle = None
                        
                        for vehicle_name in vehicle_list:
                            vehicle_loc = self.unrealcv.get_obj_location(vehicle_name)
                            
                            distance = calculate_distance(obj_loc, vehicle_loc)
                            
                            if distance < min_distance:
                                min_distance = distance
                                nearest_vehicle = vehicle_name
                        vehicle = nearest_vehicle
                        loca = self.unrealcv.get_obj_location(vehicle)
                        rot = self.unrealcv.get_obj_rotation(vehicle)
                        theta = np.deg2rad(rot[1])
                        bias = [200*np.cos(theta+np.pi/2), 200*np.sin(theta+np.pi/2), 0]
                        # bias = [200*np.cos(theta), 200*np.sin(theta+np.pi/2), 0]
                        loc = [loca[i] + bias[i] for i in range(3)]
                        self.unrealcv.set_obj_location(obj, loc)
                        time.sleep(0.5)
                        cur_loc = self.unrealcv.get_obj_location(obj)
                        cur_rot = self.unrealcv.get_obj_rotation(obj)
                        time.sleep(0.5)
                        self.unrealcv.set_obj_rotation(obj, rot)
                        self.unrealcv.set_animation(obj, 'enter_vehicle')
                        time.sleep(3.0)
                        agent_type = list(obj.split('_'))[0]  # get the agent type from the object name
                        index = self.target_configs[agent_type]['name'].index(obj)
                        self.target_configs[agent_type]['start_pos'][index] = cur_loc + cur_rot
                    else:
                        warnings.warn(f"No vehicle found for {obj} to enter.reset status.")
                        status = random.choice(['stand', 'crouch', 'liedown'])
                        self.unrealcv.set_animation(obj, status)
                        agent_type = list(obj.split('_'))[0]  # get the agent type from the object name
                        index = self.target_configs[agent_type]['name'].index(obj)
                        self.target_configs[agent_type]['state'][index] = status 
                    continue
                time.sleep(3)
            time.sleep(3.0)
        print("\n agent num:", len(self.target_list))
        # distribute camera
        self.unrealcv.cam = self.unrealcv.get_camera_config()
        self.update_camera_assignments()
        time.sleep(5)
        # get state
        observations, self.obj_poses, self.img_show = self.update_observation(self.player_list, self.cam_list, self.cam_flag, self.observation_type)

        #get camera info
        # self.info['cam_info'] = self.unrealcv.get_cam_info(self.agents[self.player_list[self.protagonist_id]]['cam_id'])
        # print(self.info['cam_info'])
        print("agent num:", len(self.target_list))
        return observations,self.info

    def update_camera_assignments(self):
        """
        更新所有智能体的相机分配，确保每个智能体使用最近的相机
        """
        # 获取所有相机位置
        cam_locs = []
        for cam_id in range(0,self.unrealcv.get_camera_num()):
            print(cam_id)
            cam_loc = self.unrealcv.get_cam_location(cam_id)
            cam_locs.append(cam_loc)

        # 为每个智能体匹配最近相机并更新
        for obj in self.player_list:
            # 获取智能体位置 (包含位置和旋转信息)
            obj_loc = self.unrealcv.get_obj_location(obj)

            dis_list = []
            for loc in cam_locs:
                # 计算距离 (使用3D欧氏距离)
                distance = self.unrealcv.get_distance(loc, obj_loc, 3)
                dis_list.append(distance)

            # 找到最小距离的索引 (即相机ID)
            nearest_cam_id = dis_list.index(min(dis_list))

            # 更新智能体的相机ID
            self.agents[obj]['cam_id'] = nearest_cam_id
            # 更新cam_list中对应的相机ID
            agent_idx = self.player_list.index(obj)
            self.cam_list[agent_idx] = nearest_cam_id

    def get_ue_pid(self):
        """
        Get the process ID of the Unreal Engine binary.

        Returns:
            int: Process ID of the Unreal Engine binary.
        """
        return self.ue_binary.ue_pid


    def close(self):
        """
        Close the environment and disconnect from UnrealCV.
        """
        if self.launched:
            self.unrealcv.client.disconnect()
            self.ue_binary.close()

    def render(self, mode='rgb_array', close=False):
        """
        Show the rendered image.

        Args:
            mode (str): Mode of rendering.
            close (bool): Flag to close the rendering.

        Returns:
            np.array: Image to be rendered.
        """
        if close==True:
            self.ue_binary.close()
        return self.img_show

    def seed(self, seed=None):
        """
        Set the random seed for the environment.

        Args:
            seed (int): Seed value.
        """
        np.random.seed(seed)

    def update_observation(self, player_list, cam_list, cam_flag, observation_type):
        """
        Update the observations for the agents.

        Args:
            player_list (list): List of player agents.
            cam_list (list): List of camera IDs.
            cam_flag (list): List of camera flags.
            observation_type (str): Type of observation.

        Returns:
            tuple: Updated observations, object poses, and image to show.
        """
        obj_poses, cam_poses, imgs, masks, depths = self.unrealcv.get_pose_img_batch(player_list, cam_list, cam_flag)
        observations = self.prepare_observation(observation_type, imgs, masks, depths, obj_poses)
        img_show = self.prepare_img2show(self.protagonist_id, observations)
        return observations, obj_poses, img_show

    def get_start_area(self, safe_start, safe_range):
        """
        Get the start area for the agents.

        Args:
            safe_start (list): Safe start coordinates.
            safe_range (int): Safe range value.

        Returns:
            list: Start area coordinates.
        """
        start_area = [safe_start[0]-safe_range, safe_start[0]+safe_range,
                     safe_start[1]-safe_range, safe_start[1]+safe_range]
        return start_area

    def set_topview(self, current_pose, cam_id):
        """
        Set the virtual camera on top of a point(current pose) to capture images from the bird's eye view.

        Args:
            current_pose (list): Current pose of the camera.
            cam_id (int): Camera ID.
        """
        cam_loc = current_pose[:3]
        cam_loc[-1] = self.height_top_view
        cam_rot = [-90, 0, 0]
        self.unrealcv.set_cam_location(cam_id, cam_loc)
        self.unrealcv.set_cam_rotation(cam_id, cam_rot)

    def get_relative(self, pose0, pose1):  # pose0-centric
        """
        Get the relative pose between two objects, pose0 is the reference object.

        Args:
            pose0 (list): Pose of the reference object (the center of the coordinate system).
            pose1 (list): Pose of the target object.

        Returns:
            tuple: Relative observation vector, distance, and angle.
        """
        delt_yaw = pose1[4] - pose0[4]
        angle = misc.get_direction(pose0, pose1)
        distance = self.unrealcv.get_distance(pose1, pose0, 3)
        obs_vector = [np.sin(delt_yaw/180*np.pi), np.cos(delt_yaw/180*np.pi),
                      np.sin(angle/180*np.pi), np.cos(angle/180*np.pi),
                      distance]
        return obs_vector, distance, angle

    def prepare_observation(self, observation_type, img_list, mask_list, depth_list, pose_list):
        """
        Prepare the observation based on the observation type.

        Args:
            observation_type (str): Type of observation.
            img_list (list): List of images.
            mask_list (list): List of masks.
            depth_list (list): List of depth images.
            pose_list (list): List of poses.

        Returns:
            np.array: Prepared observation.
        """
        if observation_type == 'Depth':
            return np.array(depth_list)
        elif observation_type == 'Mask':
            return np.array(mask_list)
        elif observation_type == 'Color':
            return np.array(img_list)
        elif observation_type == 'Rgbd':
            depth_list = np.squeeze(depth_list,axis=-1)
            return np.append(np.array(img_list), np.array(depth_list), axis=-1)
        elif observation_type == 'Pose':
            return np.array(pose_list)
        elif observation_type == 'MaskDepth':
            return np.append(np.array(mask_list), np.array(depth_list), axis=-1)
        elif observation_type =='ColorMask':
            return np.append(np.array(img_list), np.array(mask_list), axis=-1)

    def remove_config_agent(self, obj_name, agent_type):
        if hasattr(self, 'target_configs') and agent_type is not None:
            if agent_type in self.target_configs:
                config = self.target_configs[agent_type]
                index_to_remove = None
                if 'name' in config and obj_name in config['name']:
                    index_to_remove = config['name'].index(obj_name)
                
                if index_to_remove is not None:
                    for key in ['name', 'app_id', 'animation', 'start_pos']:
                        if key in config and index_to_remove < len(config[key]):
                            config[key].pop(index_to_remove)
                    
                    if len(config.get('name', [])) == 0:
                        self.target_configs.pop(agent_type)
                        print(f"having deleted type:{agent_type} in target_configs: ")

    def check_pos_dis(self, pre_pos, cur_pos, agent_type):
        error = np.array(pre_pos[:2]) - np.array(cur_pos[:2])
        distance = np.linalg.norm(error)
        if agent_type == 'player':
            return distance > 200
        else:
            return distance > 200

    def check_agent(self):
        """
        Check if the agent is in right position
        """
        target_list_copy = self.target_list.copy()
        for obj in self.target_list.copy():
            agent_type = self.target_agents[obj]['agent_type']
            cur_pos = self.unrealcv.get_obj_location(obj) + self.unrealcv.get_obj_rotation(obj)
            if self.agents[obj]['cam_id'] == -1 or self.check_pos_dis(self.target_agents[obj]['start_pos'], cur_pos, agent_type):
                print(f"deleting agent:{obj}, type:{agent_type}")
                self.remove_agent(obj)
                # self.agremove_agent_configs
                self.remove_config_agent(obj,agent_type)
        if self.is_eval:
            assert len(self.target_list) == len(target_list_copy), "Some agents are removed during evaluation!"             

        return self.target_list

    def check_cam(self, ):
        """
        Check the camera status and update the camera list.
        Assigns the closest available camera to each object.

        Returns:
            tuple: (list of camera IDs for players, list of camera IDs for targets)
        """
        new_player_cam_list = []
        # Ensure available_cam contains unique, valid camera IDs
        # Using set to get unique IDs then converting back to list
        unique_cam_ids = set(self.cam_list)
        available_cameras = [cam_id for cam_id in unique_cam_ids if cam_id >= 0]

        if not available_cameras:
            print("Warning: check_cam - no available cameras (cam_id >= 0).")
            # Assign -1 to all agents if no cameras are available
            for obj_name in self.player_list:
                if obj_name in list(self.agents.keys()):
                    self.agents[obj_name]['cam_id'] = -1
            # Return empty lists or lists of -1s based on player/target counts
            return [-1] * len(self.player_list)

        assert self.unrealcv.get_camera_num() >= len(self.player_list)
        for obj_name in self.player_list: # Renamed obj to obj_name for clarity
            if obj_name not in list(self.agents.keys()):
                print(f"Warning: check_cam - object {obj_name} not in self.agents, skipping.")
                # Decide how to handle this: append a placeholder or skip
                if obj_name in self.player_list:
                    new_player_cam_list.append(-1) # Placeholder for missing agent
                continue

            obj_loca = self.unrealcv.get_obj_location(obj_name)
            obj_rota = self.unrealcv.get_obj_rotation(obj_name) # Rotation not needed for distance

            if obj_loca is None or obj_rota is None:
                print(f"Warning: check_cam - can't get the location of {obj_name}.")
                self.agents[obj_name]['cam_id'] = -1 # Assign invalid cam_id
                if obj_name in self.player_list:
                    new_player_cam_list.append(-1)
                continue
            
            # obj_pos_for_dist = np.array(obj_loca[:3]) # We only need x,y,z for distance

            min_distance_found = float('inf') # Use a more descriptive name
            assigned_cam_id_for_obj = -1      # Store the ID of the closest camera for the current obj

            print("available_cameras:",available_cameras)
            for current_iter_cam_id in available_cameras: # Use a different loop variable name
                print("checking cam_id:", current_iter_cam_id)
                cam_loca = self.unrealcv.get_cam_location(current_iter_cam_id, syns=False)
                time.sleep(0.05) 
                cam_rota = self.unrealcv.get_cam_rotation(current_iter_cam_id, syns=False)
                time.sleep(0.05)
                if not isinstance(cam_loca, list) or not isinstance(cam_rota, list):
                    print(f"Warning: check_cam - Unable to get the position of camera {current_iter_cam_id}.")
                    continue
                
                # cam_pos_for_dist = np.array(cam_loca[:3])
                
                # Calculate distance using only the 3D location vectors
                current_cam_distance = np.linalg.norm(np.array(obj_loca+obj_rota) - np.array(cam_loca+cam_rota))

                if current_cam_distance < min_distance_found:
                    min_distance_found = current_cam_distance
                    assigned_cam_id_for_obj = current_iter_cam_id # Correctly store the ID of this closest camera
            
            # After checking all available cameras for the current obj_name:
            self.agents[obj_name]['cam_id'] = assigned_cam_id_for_obj # Update agent's cam_id

            if obj_name in self.player_list:
                new_player_cam_list.append(assigned_cam_id_for_obj)
            if assigned_cam_id_for_obj != -1:
                print(f"Object {obj_name} assigned to camera {assigned_cam_id_for_obj} (distance {min_distance_found:.2f})")
            else:
                print(f"Warning: Object {obj_name} could not be assigned a valid camera (possibly failed to get positions for all available cameras).")


        # It's generally a good practice for self.cam_list to reflect the cameras used by self.player_list
        # And self.target_camera_list for self.target_list
        # The calling function (e.g., reset) should decide if it wants to update these instance variables
        # self.cam_list = new_player_cam_list
        # self.target_camera_list = new_target_cam_list
        
        return new_player_cam_list


    def rotate2exp(self, yaw_exp, obj, th=1):
        """
        Rotate the object to the expected yaw.

        Args:
            yaw_exp (float): Expected yaw.
            obj (str): Object name.
            th (int): Threshold value.

        Returns:
            float: Delta yaw.
        """
        yaw_pre = self.unrealcv.get_obj_rotation(obj)[1]
        delta_yaw = yaw_exp - yaw_pre
        while abs(delta_yaw) > th:
            if 'Drone' in obj:
                self.unrealcv.set_move_bp(obj, [0, 0, 0, np.clip(delta_yaw, -60, 60)/60*np.pi])
            else:
                self.unrealcv.set_move_bp(obj, [np.clip(delta_yaw, -60, 60), 0])
            yaw_pre = self.unrealcv.get_obj_rotation(obj)[1]
            delta_yaw = (yaw_exp - yaw_pre) % 360
            if delta_yaw > 180:
                delta_yaw = 360 - delta_yaw
        return delta_yaw

    def relative_metrics(self, relative_pose):
        """
        Compute the relative metrics among agents for rewards and evaluation.

        Args:
            relative_pose (np.array): Relative pose array.

        Returns:
            dict: Dictionary containing collision and average distance metrics.
        """
        info = dict()
        relative_dis = relative_pose[:, :, 0]
        relative_ori = relative_pose[:, :, 1]
        collision_mat = np.zeros_like(relative_dis)
        collision_mat[np.where(relative_dis < 100)] = 1
        collision_mat[np.where(np.fabs(relative_ori) > 45)] = 0  # collision should be at the front view
        info['collision'] = collision_mat
        info['dis_ave'] = relative_dis.mean() # average distance among players, regard as a kind of density metric

        return info

    def add_agent(self, name, loc, refer_agent):
        """
        Add a new agent to the environment.

        Args:
            name (str): Name of the new agent.
            loc (list): Location of the new agent.
            refer_agent (dict): Reference agent configuration.

        Returns:
            dict: New agent configuration.
        """
        new_dict = refer_agent.copy()
        cam_num = self.unrealcv.get_camera_num()
        print(f"cam num before set:{cam_num}")
        #self.unrealcv.new_obj(refer_agent['class_name'], name, random.sample(self.safe_start, 1)[0])
        assert len(loc) == 6
        loca = []
        rot = []
        if len(loc) == 6:
            loca = loc[0:3]
            rot = loc[3:]
        elif len(loc) == 3:
            loca = loc
            rot = [0, 0, 0]
        print(f"set obj:{name}")
        cur_agent_type = refer_agent['agent_type']    
        self.unrealcv.new_obj(refer_agent['class_name'], name, loca, rot)
        print(f"obj set")

        # time.sleep(1)

        new_dict["name"] = name
        self.target_list.append(name) # record the agent I add
        self.player_list.append(name)

        if self.unrealcv.get_camera_num() > cam_num:
            new_dict['cam_id'] = cam_num
        else:
            new_dict['cam_id'] = -1
        # agent_type


        # self.target_camera_list.append(new_dict['cam_id'])
        self.cam_list.append(new_dict['cam_id'])
        time.sleep(0.5)
        self.unrealcv.set_obj_scale(name, refer_agent['scale'])
        time.sleep(0.5)
        self.unrealcv.set_obj_color(name, np.random.randint(0, 255, 3))
        time.sleep(0.5)
        self.unrealcv.set_random(name, 0)
        time.sleep(0.5)
        self.unrealcv.set_interval(self.interval, name)
        time.sleep(0.5)
        self.unrealcv.set_obj_location(name, loca)
        # for obj in self.player_list:
        #     print("num of agent:",len(self.player_list)-1)
        #     print("location of obj:", obj, "is", self.unrealcv.get_obj_location(obj))
        # if cur_agent_type in ['animal','car']:
        #     self.unrealcv.set_phy(name, 1)
        time.sleep(1)
        self.unrealcv.set_phy(name, 0)
        time.sleep(1)
        self.unrealcv.set_obj_rotation(name, rot)
        time.sleep(2)
        if cur_agent_type not in ['car','motorbike']:
            self.unrealcv.set_appearance(name, self.target_agents[name]['app_id'])
        # transform action space
        action_spaces = [self.define_action_space(self.action_type, agent_info=self.agents[obj]) 
                                for obj in self.player_list[:-1]]
        action_spaces.append(self.define_action_space(self.action_type, agent_info=new_dict))
        self.action_space = spaces.Tuple(action_spaces)
        obs_spaces = [self.define_observation_space(self.cam_list[i], self.observation_type, self.resolution) 
                                for i in range(len(self.player_list))]
        self.observation_space = spaces.Tuple(obs_spaces)
        # self.action_space.append(self.define_action_space(self.action_type, agent_info=new_dict))
        # self.observation_space.append(self.define_observation_space(new_dict['cam_id'], self.observation_type, self.resolution))
        
        return new_dict

    def remove_agent(self, name):
        """
        Remove an agent from the environment.

        Args:
            name (str): Name of the agent to be removed.
        """
        print(f'remove {name}')
        # agent_list = self.player_list + self.target_list
        agent_index = self.player_list.index(name)
        last_camera = self.cam_list

        self.player_list.remove(name)
        if name in self.target_list:
            # print("agent index:", agent_index)
            target_index = self.target_list.index(name)
            self.target_list.remove(name)
            # self.safe_start.pop(agent_index)  # remove the safe start point of the agent   
            self.target_start.pop(target_index) # 1 for the player     
        self.cam_list = self.remove_cam(name)    
        action_spaces = [self.define_action_space(self.action_type, agent_info=self.agents[obj]) 
                for obj in self.player_list]
        self.action_space = spaces.Tuple(action_spaces)
        obs_spaces = [self.define_observation_space(self.cam_list[i], self.observation_type, self.resolution) 
                    for i in range(len(self.player_list))]
        self.observation_space = spaces.Tuple(obs_spaces)
        # self.action_space.pop(agent_index)
        # self.observation_space.pop(agent_index)
        self.unrealcv.destroy_obj(name)  # the agent is removed from the scene
        self.agents.pop(name)
        time.sleep(1)
        print(f"cam num after remove:{self.unrealcv.get_camera_num()}", "length of cam list:", len(self.cam_list))
        while self.unrealcv.get_camera_num() > len(last_camera) + 1:
            print("waiting for removing camera!")
            time.sleep(1)
            pass

    # def remove_target_cam(self, name):
    #     """
    #     Remove the camera associated with an agent.

    #     Args:
    #         name (str): Name of the agent.

    #     Returns:
    #         list: Updated list of camera IDs.
    #     """
    #     cam_id = self.agents[name]['cam_id']
    #     cam_list = []
    #     for obj in self.target_list:
    #         if self.agents[obj]['cam_id'] > cam_id and cam_id > 0:
    #             self.agents[obj]['cam_id'] -= 1
    #         cam_list.append(self.agents[obj]['cam_id'])
    #     return cam_list

    def remove_cam(self, name):
        """
        Remove the camera associated with an agent.

        Args:
            name (str): Name of the agent.

        Returns:
            list: Updated list of camera IDs.
        """
        cam_id = self.agents[name]['cam_id']
        cam_list = []
        for obj in self.player_list:
            if self.agents[obj]['cam_id'] > cam_id and cam_id > 0:
                self.agents[obj]['cam_id'] -= 1
            cam_list.append(self.agents[obj]['cam_id'])
        return cam_list

    def define_action_space(self, action_type, agent_info):
        """
        Define the action space for an agent.

        Args:
            action_type (str): Type of action space ('Discrete', 'Continuous', 'Mixed').
            agent_info (dict): Agent configuration.

        Returns:
            gym.Space: Defined action space.
        """

        if action_type == 'Discrete':
            return spaces.Discrete(len(agent_info["move_action"]))
        elif action_type == 'Continuous':
            return spaces.Box(low=np.array(agent_info["move_action_continuous"]['low']),
                              high=np.array(agent_info["move_action_continuous"]['high']), dtype=np.float32)
        else:  # Hybrid
            move_space = spaces.Box(low=np.array(agent_info["move_action_continuous"]['low']),
                                    high=np.array(agent_info["move_action_continuous"]['high']), dtype=np.float32)
            turn_space = spaces.Discrete(2)
            animation_space = spaces.Discrete(2)
            if "head_action" in agent_info.keys():
                turn_space = spaces.Discrete(len(agent_info["head_action"]))
            if "animation_action" in agent_info.keys():
                animation_space = spaces.Discrete(len(agent_info["animation_action"]))
            return spaces.Tuple((move_space, turn_space, animation_space))

    def define_observation_space(self, cam_id, observation_type, resolution=(160, 120)):
        """
        Define the observation space for an agent.

        Args:
            cam_id (int): Camera ID.
            observation_type (str): Type of observation space.
            resolution (tuple): Resolution of the observation space.

        Returns:
            gym.Space: Defined observation space.
        """
        if observation_type == 'Pose' or cam_id < 0:
            observation_space = spaces.Box(low=-100, high=100, shape=(6,),
                                               dtype=np.float32)  # TODO check the range and shape
        else:
            if observation_type == 'Color' or observation_type == 'CG' or observation_type == 'Mask':
                img_shape = (resolution[1], resolution[0], 3)
                observation_space = spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8)
            elif observation_type == 'Depth':
                img_shape = (resolution[1], resolution[0], 1)
                observation_space = spaces.Box(low=0, high=100, shape=img_shape, dtype=np.float32)
            elif observation_type == 'Rgbd':
                s_low = np.zeros((resolution[1], resolution[0], 4))
                s_high = np.ones((resolution[1], resolution[0], 4))
                s_high[:, :, -1] = 100.0  # max_depth
                s_high[:, :, :-1] = 255  # max_rgb
                observation_space = spaces.Box(low=s_low, high=s_high, dtype=np.float32)
            elif observation_type == 'MaskDepth':
                s_low = np.zeros((resolution[1], resolution[0], 4))
                s_high = np.ones((resolution[1], resolution[0], 4))
                s_high[:, :, -1] = 100.0  # max_depth
                s_high[:, :, :-1] = 255  # max_rgb
                observation_space = spaces.Box(low=s_low, high=s_high, dtype=np.float32)
            elif observation_type=='ColorMask':
                img_shape = (resolution[1], resolution[0], 6)
                observation_space = spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8)
        return observation_space

    def sample_init_pose(self, use_reset_area=False, num_agents=1):
        """
        Sample initial poses to reset the agents.

        Args:
            use_reset_area (bool): Flag to indicate whether to use the reset area for sampling.
            num_agents (int): Number of agents to sample poses for.

        Returns:
            list: List of sampled locations for the agents.
        """
        if num_agents > len(self.safe_start):
            use_reset_area = True
            warnings.warn('The number of agents is less than the number of pre-defined start points, random sample points from the pre-defined area instead.')
        if use_reset_area:
            locations = self.sample_from_area(self.reset_area, num_agents)  # sample from a pre-defined area
            #locations = self.resert_area
        else:
            #locations = random.sample(self.safe_start, num_agents) # sample one pre-defined start point
            locations = self.safe_start
        #print(locations)
        return locations

    def random_app(self):
        """
        Randomly assign an appearance to each agent in the player list based on their category.

        The appearance is selected from a predefined range of IDs for each category.

        Categories:
            - player: IDs from 1 to 18
            - animal: IDs from 0 to 26
        """
        app_map = {
            'player': range(1, 19),
            'animal': range(0, 27),
            'drone':range(0,1)
        }
        for obj in self.player_list:
            category = self.agents[obj]['agent_type']
            if category not in app_map.keys():
                continue
            app_id = np.random.choice(app_map[category])
            self.unrealcv.set_appearance(obj, app_id)

    def environment_augmentation(self, player_mesh=False, player_texture=False,
                                 light=False, background_texture=False,
                                 layout=False, layout_texture=False):
        """
        Randomly assign an appearance to each agent in the player list based on their category.

        The appearance is selected from a predefined range of IDs for each category.

        Categories:
            - player: IDs from 1 to 18
            - animal: IDs from 0 to 26
        """
        app_map = {
            'player': range(1, 19),
            'animal': range(0, 27),
            'drone': range(0,1)
        }
        if player_mesh:  # random human mesh
            for obj in self.player_list:
                app_id = np.random.choice(app_map[self.agents[obj]['agent_type']])
                self.unrealcv.set_appearance(obj, app_id)
        # random light and texture of the agents
        if player_texture:
            if self.env_name == 'MPRoom':  # random target texture
                for obj in self.player_list:
                    if self.agents[obj]['agent_type'] == 'player':
                        self.unrealcv.random_player_texture(obj, self.textures_list, 3)
        if light:
            self.unrealcv.random_lit(self.env_configs["lights"])

        # random the texture of the background
        if background_texture:
            self.unrealcv.random_texture(self.env_configs["backgrounds"], self.textures_list, 5)

        # random place the obstacle`
        if layout:
            self.unrealcv.clean_obstacles()
            self.unrealcv.random_obstacles(self.objects_list, self.textures_list,
                                           len(self.objects_list), self.reset_area, self.start_area, layout_texture)

    def get_pose_states(self, obj_pos):
        # get the relative pose of each agent and the absolute location and orientation of the agent
        pose_obs = []
        player_num = len(obj_pos)
        np.zeros((player_num, player_num, 2))
        relative_pose = np.zeros((player_num, player_num, 2))
        for j in range(player_num):
            vectors = []
            for i in range(player_num):
                obs, distance, direction = self.get_relative(obj_pos[j], obj_pos[i])
                yaw = obj_pos[j][4]/180*np.pi
                # rescale the absolute location and orientation
                abs_loc = [obj_pos[i][0], obj_pos[i][1],
                           obj_pos[i][2], np.cos(yaw), np.sin(yaw)]
                obs = obs + abs_loc
                vectors.append(obs)
                relative_pose[j, i] = np.array([distance, direction])
            pose_obs.append(vectors)

        return np.array(pose_obs), relative_pose

    def launch_ue_env(self, ue_log_path=None):
        # launch the UE4 binary
        env_ip, env_port = self.ue_binary.start(docker=self.docker, resolution=self.resolution, display=self.display,
                                               opengl=self.use_opengl, offscreen=self.offscreen_rendering,
                                               nullrhi=self.nullrhi,sleep_time=10,log_file_path=ue_log_path)


        # connect to UnrealCV Server
        self.unrealcv = Character_API(port=env_port, ip=env_ip, resolution=self.resolution, comm_mode=self.comm_mode)
        self.unrealcv.set_map(self.env_name)

        # send pid if needed
        if self.send_pid:
            ue_pid = self.get_ue_pid()
            send_pid(pid=ue_pid, port=self.watchdog_port)
            print("ue pid:", ue_pid)
        return True

    def init_agents(self):
        print("safe start:", self.safe_start)
        for obj in self.player_list.copy(): # the agent will be fully removed in self.agents
            if self.agents[obj]['agent_type'] not in self.agents_category or obj in self.target_list:
                self.remove_agent(obj)
            # self.target_list = []
        # destory all pickup objects
        if len(self.pickup_list) != 0:
            for obj in self.pickup_list.copy():
                self.unrealcv.destroy_obj(obj)
                self.pickup_list.remove(obj)
        assert len(self.pickup_list) == 0, "There are still some pickup objects in the scene, please check the code."

        for obj in self.player_list:
            self.unrealcv.set_obj_scale(obj, self.agents[obj]['scale'])
            self.unrealcv.set_random(obj, 0)
            self.unrealcv.set_interval(self.interval, obj)
            #self.unrealcv.set_animation(obj, 'crouch')
        self.agent_color_dict = self.unrealcv.build_color_dict(self.player_list)
        self.cam_flag = self.get_cam_flag(self.observation_type)


    def init_objects(self):
        self.unrealcv.init_objects(self.objects_list)

    def prepare_img2show(self, index, states):
        if self.observation_type == 'Rgbd':
            return states[index][:, :, :3]
        elif self.observation_type in ['Color', 'Gray', 'CG', 'Mask']:
            return states[index]
        elif self.observation_type == 'Depth':
            return states[index]/states[index].max()  # normalize the depth image
        else:
            return None

    def set_population(self, num_agents):
        agent_cnt = 0

        if len(self.player_list) < num_agents:
            self.valid_targets = [agent_name for agent_name in self.target_agents.keys() 
                       if self.target_agents[agent_name]['agent_type'] in self.refer_agents_category]
            valid_agents = self.valid_targets.copy()
            if not valid_agents:
                raise ValueError(f"No valid agent to refer, please check the {self.refer_agents_category}")
            for valid_name in valid_agents:
                # self.cur_agent_type = self.target_agents[valid_name]['agent_type']
                refer_agent = self.refer_agents[self.agent_configs[self.target_agents[valid_name]["agent_type"]]["name"][0]]
                if refer_agent['agent_type'] == 'car':
                    refer_agent['class_name'] = self.target_agents[valid_name]['type']
                self.target_start.append(self.target_agents[valid_name]["start_pos"])
                self.agents[f'{valid_name}'] = self.add_agent(valid_name, self.target_agents[valid_name]["start_pos"], refer_agent)
                agent_cnt += 1
                time.sleep(1)
        while len(self.player_list) > num_agents:
            self.remove_agent(self.player_list[-1])  # remove the last one
    

    def set_npc(self):
        # TODO: set the NPC agent
        return self.player_list.index(random.choice([x for x in self.player_list if x > 0]))

    def set_agent(self):
        # the agent is controlled by the external controller
        return self.cam_list.index(random.choice([x for x in self.cam_list if x > 0]))

    def action_mapping(self, actions, player_list):
        actions2move = []
        actions2animate = []
        actions2head = []
        actions2player = []
        for i, obj in enumerate(player_list):
            action_space = self.action_space[i]
            act = actions[i]
            if act is None:  # if the action is None, then we don't control this agent
                actions2move.append(None)  # place holder
                actions2animate.append(None)
                actions2head.append(None)
                continue
            if isinstance(action_space, spaces.Discrete):
                actions2move.append(self.agents[obj]["move_action"][act])
                actions2animate.append(None)
                actions2head.append(None)
            elif isinstance(action_space, spaces.Box):
                actions2move.append(act)
                actions2animate.append(None)
                actions2head.append(None)
            elif isinstance(action_space, spaces.Tuple):
                for j, action in enumerate(actions[i]):
                    if j == 0:
                        if isinstance(action, int):
                            actions2move.append(self.agents[obj]["move_action"][action])
                        else:
                            actions2move.append(action)
                    elif j == 1:
                        if isinstance(action, int):
                            actions2head.append(self.agents[obj]["head_action"][action])
                        else:
                            actions2head.append(action)
                    elif j == 2:
                        actions2animate.append(self.agents[obj]["animation_action"][action])
            
        return actions2move, actions2head, actions2animate


    def get_cam_flag(self, observation_type, use_color=False, use_mask=False, use_depth=False, use_cam_pose=False):
        # get flag for camera
        # observation_type: 'color', 'depth', 'mask', 'cam_pose'
        flag = [False, False, False, False]
        flag[0] = use_cam_pose
        flag[1] = observation_type == 'Color' or observation_type == 'Rgbd' or use_color or observation_type == 'ColorMask'
        flag[2] = observation_type == 'Mask' or use_mask or observation_type == 'MaskDepth' or observation_type == 'ColorMask'
        flag[3] = observation_type == 'Depth' or observation_type == 'Rgbd' or use_depth or observation_type == 'MaskDepth'
        print('cam_flag:', flag)
        return flag

    def sample_from_area(self, area, num):
        x = np.random.randint(area[0], area[1], num)
        y = np.random.randint(area[2], area[3], num)
        z = np.random.randint(area[4], area[5], num)
        return np.vstack((x, y, z)).T

    def get_startpoint(self, target_pos=[], distance=None, reset_area=[], exp_height=200, direction=None):
        for i in range(5):  # searching a safe point
            if direction == None:
                direction = 2 * np.pi * np.random.sample(1)
            else:
                direction = direction % (2 * np.pi)
            if distance == None:
                x = np.random.randint(reset_area[0], reset_area[1])
                y = np.random.randint(reset_area[2], reset_area[3])
            else:
                dx = float(distance * np.cos(direction))
                dy = float(distance * np.sin(direction))
                x = dx + target_pos[0]
                y = dy + target_pos[1]
            cam_pos_exp = [x, y, exp_height]
            if reset_area[0] < x < reset_area[1] and reset_area[2] < y < reset_area[3]:
                cam_pos_exp[0] = x
                cam_pos_exp[1] = y
                return cam_pos_exp
        return []