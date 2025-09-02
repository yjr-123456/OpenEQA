import warnings
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gym_unrealcv.envs.utils import misc
from unrealcv.launcher import RunUnreal
from gym_unrealcv.envs.agent.character import Character_API
import random
import sys
''' 
It is a base env for general purpose agent-env interaction, including single/multi-agent navigation, tracking, etc.
Observation : raw color image and depth
Action:  Discrete/Continuous
Done : define by the task wrapper
'''

# TODO: agent apis for blueprints commands
# TODO: config env by parapmeters
# TODO: matain a general agent list
class UnrealCv_Random_base(gym.Env):
    def __init__(self,
                 setting_file,  # the setting file to define the task
                 action_type='Discrete',  # 'discrete', 'continuous'
                 observation_type='Color',  # 'color', 'depth', 'rgbd', 'Gray'
                 resolution=(160, 160),
                 reset_type = 0,
                 reward_type = 'distance',
                 docker = False
                 ):

        setting = misc.load_env_setting(setting_file)
        self.env_name = setting['env_name']
        self.max_steps = setting['max_steps'] if 'max_steps' in setting else 100
        self.height = setting['height']
        self.cam_id = [setting['third_cam']['cam_id']]
        self.agent_configs = setting['agents']
        self.player_move_action = setting['agents']['player']['move_action']
        self.sample_region = setting['sample_region'] if 'sample_region' in setting else None
        self.env_configs = setting["env"]
        self.agents = misc.convert_dict(self.agent_configs)
        self.reset_type = reset_type
        self.reward_type = reward_type
        # TODO: it is useless.
        self.character = {
            'player': [], # the list of player to control
            'npc': [], # the list of Non-player character
            'freeze': [], # the list of player that exists in the scene, but it is frozen
        }

        self.height_top_view = setting['third_cam']['height_top_view']

        # self.env_obj_list = self.env_configs[""]
        self.objects_list = []
        self.reset_area = setting['reset_area']

        self.safe_start = setting['safe_start']
        self.interval = setting['interval']
        self.random_init = setting['random_init']
        self.start_area = self.get_start_area(self.safe_start[0], 500) # the start area of the agent, where we don't put obstacles

        self.count_eps = 0
        self.count_steps = 0

        # env configs
        self.docker = False
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
        self.protagonist_id = 0
        # self.protagonist_id = 1 

        # init agents
        self.player_list = list(self.agents.keys())
        self.cam_list = [self.agents[player]['cam_id'] for player in self.player_list]

        # define action space
        self.action_type = action_type
        assert self.action_type in ['Discrete', 'Continuous', 'Mixed']
        self.action_space = [self.define_action_space(self.action_type, self.agents[obj]) for obj in self.player_list]
        self.action_clip = None

        # define observation space,
        # color, depth, rgbd,...
        self.observation_type = observation_type
        assert self.observation_type in ['Color', 'Depth', 'Rgbd', 'Gray', 'CG', 'Mask', 'Pose','MaskDepth','ColorMask']
        self.observation_space = [self.define_observation_space(self.cam_list[i], self.observation_type, resolution)
                                  for i in range(len(self.player_list))]

        self.info = dict(
            Collision=0,
            Done=False,
            Reward=0.0,
            Action=[0,0],
            Pose=[],
            Steps=self.count_steps,
            Direction=None,
            Distance=None,
            Color=None,
            Depth=None,
            Relative_Pose=[],
            Truncation = False,
            Termination = False
        )

        # config unreal env
        if 'linux' in sys.platform:
            env_bin = setting['env_bin']
        elif 'win' in sys.platform:
            env_bin = setting['env_bin_win']
        if 'env_map' in setting.keys():
            env_map = setting['env_map']
        else:
            env_map = None

        self.ue_binary = RunUnreal(ENV_BIN=env_bin, ENV_MAP=env_map)

    def step(self, actions):
        self.info = dict(
            Collision=0,
            Done=False,
            Reward=0.0,
            Action=actions,
            Pose=[],
            Steps=self.count_steps,
            Direction=None,
            Distance=None,
            Color=None,
            Depth=None,
            Relative_Pose=[],
            Truncation = False,
            Termination = False
        )

        actions2move, actions2turn, actions2animate = self.action_mapping(actions, self.player_list)
        move_cmds = [self.unrealcv.set_move_bp(obj, actions2move[i], return_cmd=True) for i, obj in enumerate(self.player_list) if actions2move[i] is not None]
        head_cmds = [self.unrealcv.set_cam(obj, self.agents[obj]['relative_location'], actions2turn[i], return_cmd=True) for i, obj in enumerate(self.player_list) if actions2turn[i] is not None]
        anim_cmds = [self.unrealcv.set_animation(obj, actions2animate[i], return_cmd=True) for i, obj in enumerate(self.player_list) if actions2animate[i] is not None]
        self.unrealcv.batch_cmd(move_cmds+head_cmds+anim_cmds, None)
        self.count_steps += 1

        # get states
        # obj_poses, cam_poses, imgs, masks, depths = self.unrealcv.get_pose_img_batch(self.player_list, self.cam_list, self.cam_flag)
        
        # #only get controlled agent observation
        # # obj_poses, cam_poses, imgs, masks, depths = self.unrealcv.get_pose_img_batch([self.player_list[self.protagonist_id]], [self.cam_list[self.protagonist_id]], self.cam_flag)

        # self.obj_poses = obj_poses
        # # observations = self.prepare_observation(self.observation_type, imgs, masks, depths, obj_poses)
        # # self.img_show = self.prepare_img2show(self.protagonist_id, observations)
        # observations = None

        # pose_obs, relative_pose = self.get_pose_states(obj_poses)

        # # prepare the info
        # self.info['Pose'] = obj_poses
        # self.info['Relative_Pose'] = relative_pose
        # self.info['Pose_Obs'] = pose_obs
        # self.info['Reward'] = np.zeros(len(self.player_list))

        if self.count_steps > self.max_steps:
            self.info['Termination'] = True
        
        observations = None
        return observations, self.info['Reward'], self.info['Termination'], self.info['Truncation'], self.info

    def reset(self,seed=None, options=None):
        super().reset(seed=seed)
        if not self.launched:  # first time to launch
            self.launched = self.launch_ue_env()
            self.init_agents()
            self.init_objects()

        self.count_close = 0
        self.count_steps = 0
        self.count_eps += 1

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
                self.unrealcv.set_phy(obj, 0)

        # reset target location
        
        init_poses = self.sample_init_pose(self.random_init, len(self.player_list))
        for i, obj in enumerate(self.player_list):
            self.unrealcv.set_obj_location(obj, init_poses[i])
            yaw = random.uniform(-180.0,180.0)
            rotation = [0,yaw,0]
            self.unrealcv.set_obj_rotation(obj, rotation)
            # set view point
            self.unrealcv.set_cam(obj, self.agents[obj]['relative_location'], self.agents[obj]['relative_rotation'])

        # get state
        # observations, self.obj_poses, self.img_show = self.update_observation(self.player_list, self.cam_list, self.cam_flag, self.observation_type)
        observations = None
        return observations, self.info

    def close(self):
        if self.launched:
            self.unrealcv.client.disconnect()
            self.ue_binary.close()

    def render(self, mode='rgb_array', close=False):
        if close==True:
            self.ue_binary.close()
        return self.img_show

    def seed(self, seed=None):
        np.random.seed(seed)
        # if seed is not None:
        #     self.player_num = seed % (self.max_player_num-2) + 2

    def update_observation(self, player_list, cam_list, cam_flag, observation_type):
        obj_poses, cam_poses, imgs, masks, depths = self.unrealcv.get_pose_img_batch(player_list, cam_list, cam_flag)
        observations = self.prepare_observation(observation_type, imgs, masks, depths, obj_poses)
        img_show = self.prepare_img2show(self.protagonist_id, observations)
        return observations, obj_poses, img_show

    def get_start_area(self, safe_start, safe_range):
        start_area = [safe_start[0]-safe_range, safe_start[0]+safe_range,
                     safe_start[1]-safe_range, safe_start[1]+safe_range]
        return start_area

    def set_topview(self, current_pose, cam_id):
        cam_loc = current_pose[:3]
        cam_loc[-1] = self.height_top_view
        cam_rot = [-90, 0, 0]
        self.unrealcv.set_cam_location(cam_id, cam_loc)
        self.unrealcv.set_cam_rotation(cam_id, cam_rot)

    def get_relative(self, pose0, pose1):  # pose0-centric
        delt_yaw = pose1[4] - pose0[4]
        angle = misc.get_direction(pose0, pose1)
        distance = self.unrealcv.get_distance(pose1, pose0, 3)
        # distance_norm = distance / self.exp_distance
        obs_vector = [np.sin(delt_yaw/180*np.pi), np.cos(delt_yaw/180*np.pi),
                      np.sin(angle/180*np.pi), np.cos(angle/180*np.pi),
                      distance]
        return obs_vector, distance, angle

    def prepare_observation(self, observation_type, img_list, mask_list, depth_list, pose_list):
        if observation_type == 'Depth':
            return np.array(depth_list)
        elif observation_type == 'Mask':
            return np.array(mask_list)
        elif observation_type == 'Color':
            return np.array(img_list)
        elif observation_type == 'Rgbd':
            # depth_list = [np.squeeze(depth, axis= 2) for depth in depth_list]
            depth_list = np.squeeze(depth_list,axis=-1)
            return np.append(np.array(img_list), np.array(depth_list), axis=-1)
        elif observation_type == 'Pose':
            return np.array(pose_list)
        elif observation_type == 'MaskDepth':
            return np.append(np.array(mask_list), np.array(depth_list), axis=-1)
        elif observation_type =='ColorMask':
            return np.append(np.array(img_list), np.array(mask_list), axis=-1)



    def rotate2exp(self, yaw_exp, obj, th=1):
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
        # compute the relative relation (collision, in-the-view, misleading) among agents for rewards and evaluation metrics
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
        new_dict = refer_agent.copy()
        cam_num = self.unrealcv.get_camera_num()
        self.unrealcv.new_obj(refer_agent['class_name'], name, random.sample(self.safe_start, 1)[0])
        self.player_list.append(name)
        if self.unrealcv.get_camera_num() > cam_num:
            new_dict['cam_id'] = cam_num
        else:
            new_dict['cam_id'] = -1
        self.cam_list.append(new_dict['cam_id'])
        self.unrealcv.set_obj_scale(name, refer_agent['scale'])
        self.unrealcv.set_obj_color(name, np.random.randint(0, 255, 3))
        self.unrealcv.set_random(name, 0)
        self.unrealcv.set_interval(self.interval, name)
        self.unrealcv.set_obj_location(name, loc)
        self.action_space.append(self.define_action_space(self.action_type, agent_info=new_dict))
        self.observation_space.append(self.define_observation_space(new_dict['cam_id'], self.observation_type, self.resolution))
        self.unrealcv.set_phy(name, 0)
        return new_dict

    def remove_agent(self, name):
        # print(f'remove {name}')
        agent_index = self.player_list.index(name)
        self.player_list.remove(name)
        self.cam_list = self.remove_cam(name)
        self.action_space.pop(agent_index)
        self.observation_space.pop(agent_index)
        self.unrealcv.destroy_obj(name)  # the agent is removed from the scene
        self.agents.pop(name)

    def remove_cam(self, name):
        cam_id = self.agents[name]['cam_id']
        cam_list = []
        for obj in self.player_list:
            if self.agents[obj]['cam_id'] > cam_id and cam_id > 0:
                self.agents[obj]['cam_id'] -= 1
            cam_list.append(self.agents[obj]['cam_id'])
        return cam_list

    def define_action_space(self, action_type, agent_info):
        if action_type == 'Discrete':
            return spaces.Discrete(len(agent_info["move_action"]))
        elif action_type == 'Continuous':
            return spaces.Box(low=np.array(agent_info["move_action_continuous"]['low']),
                              high=np.array(agent_info["move_action_continuous"]['high']), dtype=np.float32)
        else:  # Hybrid
            move_space = spaces.Box(low=np.array(agent_info["move_action_continuous"]['low']),
                                    high=np.array(agent_info["move_action_continuous"]['high']), dtype=np.float32)
            turn_space = None
            animation_space = None
            if "head_action" in agent_info.keys():
                turn_space = spaces.Discrete(len(agent_info["head_action"]))
            if "animation_action" in agent_info.keys():
                animation_space = spaces.Discrete(len(agent_info["animation_action"]))
            return spaces.Tuple((move_space, turn_space, animation_space))

    def define_observation_space(self, cam_id, observation_type, resolution=(160, 120)):
        if observation_type == 'Pose' or cam_id < 0:
            observation_space = spaces.Box(low=-100, high=100, shape=(6,),
                                               dtype=np.float16)  # TODO check the range and shape
        else:
            if observation_type == 'Color' or observation_type == 'CG' or observation_type == 'Mask':
                img_shape = (resolution[1], resolution[0], 3)
                observation_space = spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8)
            elif observation_type == 'Depth':
                img_shape = (resolution[1], resolution[0], 1)
                observation_space = spaces.Box(low=0, high=100, shape=img_shape, dtype=np.float16)
            elif observation_type == 'Rgbd':
                s_low = np.zeros((resolution[1], resolution[0], 4))
                s_high = np.ones((resolution[1], resolution[0], 4))
                s_high[:, :, -1] = 100.0  # max_depth
                s_high[:, :, :-1] = 255  # max_rgb
                observation_space = spaces.Box(low=s_low, high=s_high, dtype=np.float16)
            elif observation_type == 'MaskDepth':
                s_low = np.zeros((resolution[1], resolution[0], 4))
                s_high = np.ones((resolution[1], resolution[0], 4))
                s_high[:, :, -1] = 100.0  # max_depth
                s_high[:, :, :-1] = 255  # max_rgb
                observation_space = spaces.Box(low=s_low, high=s_high, dtype=np.float16)
            elif observation_type=='ColorMask':
                img_shape = (resolution[1], resolution[0], 6)
                observation_space = spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8)
        return observation_space

    def sample_init_pose(self, use_reset_area=False, num_agents=1):
        # sample poses to reset the agents
        if num_agents > len(self.safe_start):
            use_reset_area = True
            warnings.warn('The number of agents is less than the number of pre-defined start points, random sample points from the pre-defined area instead.')
        if use_reset_area:
            locations = self.sample_from_area(self.reset_area, num_agents)  # sample from a pre-defined area
        else:
            locations = random.sample(self.safe_start, num_agents) # sample one pre-defined start point
        return locations

    def random_app(self):
        app_map = {
            'player': range(1, 19),
            'animal': range(0, 27),
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
        app_map = {
            'player': range(1, 19),
            'animal': range(0, 27),
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

    def launch_ue_env(self):
        # launch the UE4 binary and connect to UnrealCV
        env_ip, env_port = self.ue_binary.start(docker=self.docker, resolution=self.resolution, display=self.display,
                                               opengl=self.use_opengl, offscreen=self.offscreen_rendering,
                                               nullrhi=self.nullrhi)
        # connect UnrealCV
        self.unrealcv = Character_API(port=env_port, ip=env_ip, resolution=self.resolution, comm_mode=self.comm_mode)

        return True

    def init_agents(self):
        for obj in self.player_list.copy(): # the agent will be fully removed in self.agents
            if self.agents[obj]['agent_type'] not in self.agents_category:
                self.remove_agent(obj)

        for obj in self.player_list:
            self.unrealcv.set_obj_scale(obj, self.agents[obj]['scale'])
            self.unrealcv.set_random(obj, 0)
            self.unrealcv.set_interval(self.interval, obj)

        self.unrealcv.build_color_dict(self.player_list)
        # TODO
        self.cam_flag = self.get_cam_flag(self.observation_type)
        # self.cam_flag = [True, False,False,False]

    def init_objects(self):
        self.unrealcv.init_objects(self.objects_list)

    def prepare_img2show(self, index, states):
        if self.observation_type == 'Rgbd':
            return states[index][:, :, :4]
        elif self.observation_type in ['Color', 'Gray', 'CG', 'Mask']:
            return states[index]
        elif self.observation_type == 'Depth':
            return states[index]/states[index].max()  # normalize the depth image
        else:
            return None

    def set_population(self, num_agents):
        while len(self.player_list) < num_agents:
            refer_agent = self.agents[random.choice(list(self.agents.keys()))]
            name = f'{refer_agent["agent_type"]}_EP{self.count_eps}_{len(self.player_list)}'
            self.agents[name] = self.add_agent(name, random.choice(self.safe_start), refer_agent)
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
            #discrete clip
            dis_clip = None
            action_space = self.action_space[i]
            act = actions[i]
            if self.action_clip is not None:
                dis_clip = self.action_clip[obj][act]
            if act is None:  # if the action is None, then we don't control this agent
                actions2move.append(None)  # place holder
                actions2animate.append(None)
                actions2head.append(None)
                continue
            if isinstance(action_space, spaces.Discrete):
                move = self.agents[obj]["move_action"][act]
                if dis_clip is not None:
                    move[1] = dis_clip*move[1]
                actions2move.append(move)
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