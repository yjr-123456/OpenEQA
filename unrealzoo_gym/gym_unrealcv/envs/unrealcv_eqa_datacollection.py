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
        #self.info['tsdf_info'] = self.tsdf_info
        #self.info['explore_offset'] = self.explore_offset
        self.name_mapping_dict = load_json_file("./agent_configs_sampler/agent_caption/agent_name.json")
        self.name_mapping = {}


    
    def step(self,action):
        obs, rewards, termination, truncation,info = super(UnrealCvEQA_DATA, self).step(action)
        
        return obs, rewards, termination, truncation,info
    
    def reset(self,seed=None, options=None):
        obs, info = super(UnrealCvEQA_DATA, self).reset(seed=seed, options=options)
        #obs, reward, termination, truncation, info = super(UnrealCvEQA_DATA, self).step([[0,0]])
        #get obj_names
        
        object_dict, self.info["batch_id"] = self.get_objects_info(self.target_list)
        self.info["object_dict"] = self.enhance_objects_info(object_dict)
       
        self.info["gt_information"] = self.collect_ground_truth_stats(self.info["object_dict"])
        self.info["sample_configs"] = {
            "sample_center":self.sampling_center,
            "sample_radius":self.sampling_radius
        }

        return obs, self.info
    
    def check_pos_dis(self, pre_pos, cur_pos, agent_type):
        error = np.array(pre_pos[:3]) - np.array(cur_pos[:3])
        distance = np.linalg.norm(error)
        if agent_type == 'player':
            return distance > 100
        elif agent_type in ['car', 'motorbike']:
            return distance > 200
        else:
            return distance > 100

    def remove_config_agent(self, obj_name, agent_type):
        if hasattr(self, 'target_configs') and agent_type is not None:
            if agent_type in self.target_configs:
                config = self.target_configs[agent_type]
                index_to_remove = None
                if 'name' in config and obj_name in config['name']:
                    index_to_remove = config['name'].index(obj_name)
                
                if index_to_remove is not None:
                    for key in ['name', 'app_id', 'animation', 'start_pos','feature_caption']:
                        if key in config and index_to_remove < len(config[key]):
                            config[key].pop(index_to_remove)
                    
                    if len(config.get('name', [])) == 0:
                        self.target_configs.pop(agent_type)
                        print(f"having deleted type:{agent_type} in target_configs: ")
    
    def check_agent(self):
        """
        Check if the agent is in right position
        """
        for obj in self.target_list.copy():
            agent_type = self.target_agents[obj]['agent_type']
            cur_pos = self.unrealcv.get_obj_location(obj) + self.unrealcv.get_obj_rotation(obj)
            if self.agents[obj]['cam_id'] == -1 or self.check_pos_dis(self.target_agents[obj]['start_pos'], cur_pos, agent_type):
                
                print(f"deleting agent:{obj}, type:{agent_type}")
                self.remove_agent(obj)
                # self.agent_configs
                self.remove_config_agent(obj,agent_type)
        return self.target_list
    
    def get_objects_info(self, obj_names=None):
        def get_type_id(name):
            parts = name.split('_')
            type, agent_id, batch_id = parts[0], parts[1], parts[2]
            return type, int(agent_id), batch_id

        objects_dict = {}
        self.name_mapping = {}
        if obj_names is None:
            obj_names = self.unrealcv.get_objects()
        
        for obj_name in obj_names:
            try:
                location = self.unrealcv.get_obj_location(obj_name)
                rotation = self.unrealcv.get_obj_rotation(obj_name)
                
                #color = self.unrealcv.get_obj_color(obj_name)
                type, id, batch_id = get_type_id(obj_name)
                agent_name = self.name_mapping_dict[type][id]
                assert agent_name is not None
                self.name_mapping[obj_name] = agent_name
                objects_dict[agent_name] = {
                        'location': location,
                        'rotation': rotation,
                        'state': self.target_agents[obj_name]['animation'],
                        #'color': color
                    }
            except Exception as e:
                print(f"Getting obj: {obj_name}raises an error: {e}")
        
        return objects_dict, batch_id
    
    def enhance_objects_info(self, object_dict=None):
        """
        Enhance object dictionary with natural language descriptions:
        1. Convert colors to natural language
        2. Calculate relative positions between objects
        
        Args:
            object_pitch, yaw, roll = obj1_info['rotation']dict: Dictionary of objects with their information, if None, uses self.info["object_dict"]
            
        Returns:
            Enhanced dictionary with natural language descriptions
        """
        import numpy as np
        from scipy.spatial.transform import Rotation as R
        import math

        assert object_dict is not None
        
        enhanced_dict = {}
        
        # First pass: create enhanced entries and convert colors
        for obj_name, obj_info in object_dict.items():
            # Convert numpy arrays to lists if needed
            location = obj_info['location']
            rotation = obj_info['rotation']
            state = obj_info['state']
            #color_rgb = obj_info['color']
            
            # Convert color to natural language
            #color_name = self._get_closest_color_name(color_rgb, color_names)
            
            # Create enhanced entry
            enhanced_dict[obj_name] = {
                'state': state,
                'location': location,
                'rotation': rotation,
                #'color': color_rgb,
                #'color_name': color_name,
                'relative_positions_description': {}
            }
        
        # Second pass: calculate relative positions
        for obj1_name, obj1_info in enhanced_dict.items():
            for obj2_name, obj2_info in enhanced_dict.items():
                if obj1_name != obj2_name:
                    # Get locations and rotations
                    loc1 = np.array(obj1_info['location'])
                    loc2 = np.array(obj2_info['location'])
                    
                    # Calculate vector from obj1 to obj2
                    direction_vector = loc2 - loc1
                    
                    # Get the distance
                    distance = np.linalg.norm(direction_vector)
                    
                    # Convert obj1's rotation to a rotation matrix
                    pitch, yaw, roll = obj1_info['rotation']
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
                    #direction_desc = self._get_direction_description(angle)
                    direction_desc = self._get_3d_direction_description(local_direction)
                    # Store the relative position
                    enhanced_dict[obj1_name]['relative_positions_description'][obj2_name] = {
                        # 'horizontal_angle': angle,
                        'relative_direction': direction_desc,
                        'distance': f"{distance}cm"
                        # 'description': f"The {obj2_name} is {direction_desc} the {obj1_name}, about {int(distance)} units away."
                    }
        return enhanced_dict
    
    def _get_closest_color_name(self, color_rgb, color_names):
        """Find the closest color name for an RGB value"""
        min_distance = float('inf')
        closest_name = "unknown"
        
        # Convert to numpy array for easier calculations
        color_array = np.array(color_rgb)
        
        for rgb, name in color_names.items():
            rgb_array = np.array(rgb)
            # Calculate Euclidean distance in RGB space
            distance = np.linalg.norm(color_array - rgb_array)
            if distance < min_distance:
                min_distance = distance
                closest_name = name
        
        return closest_name
    
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
            return "near"  # This shouldn't happen, but just in case
    
    def _get_3d_direction_description(self, local_direction):
        """
        Convert a 3D direction vector to a natural language description
        
        Args:
            local_direction: 3D direction vector in the object's local frame
            
        Returns:
            String describing the relative 3D position
        """
        import math
        # Calculate horizontal angle in XY plane
        horizontal_angle = math.degrees(math.atan2(local_direction[1], local_direction[0]))
        
        # Normalize horizontal angle to -180 to 180
        while horizontal_angle > 180:
            horizontal_angle -= 360
        while horizontal_angle < -180:
            horizontal_angle += 360
        
        # Calculate vertical angle (elevation)
        horizontal_distance = math.sqrt(local_direction[0]**2 + local_direction[1]**2)
        vertical_angle = math.degrees(math.atan2(local_direction[2], horizontal_distance))
        
        # Define thresholds for vertical descriptions
        high_vertical_threshold = 80  # Almost directly above/below
        vertical_threshold = 60       # Significantly above/below
        slight_vertical_threshold = 25  # Slightly above/below
        
        # Handle predominantly vertical directions
        if vertical_angle > high_vertical_threshold:
            return "directly above"
        elif vertical_angle < -high_vertical_threshold:
            return "directly below"
        
        # Get horizontal direction using existing method
        horizontal_desc = self._get_direction_description(horizontal_angle)
        
        # Combine with vertical information
        if vertical_angle > vertical_threshold:
            return f"above and {horizontal_desc}"
        elif vertical_angle > slight_vertical_threshold:
            return f"slightly above and {horizontal_desc}"
        elif vertical_angle < -vertical_threshold:
            return f"below and {horizontal_desc}"
        elif vertical_angle < -slight_vertical_threshold:
            return f"slightly below and {horizontal_desc}"
        else:
            return horizontal_desc  # Mostly horizontal

    def collect_ground_truth_stats(self, object_dict=None):
        """
        收集场景中agent的统计信息并返回结构化字典数据
        
        Args:
            object_dict (dict, optional): 可选的外部传入object字典
        
        Returns:
            dict: 包含基本信息、计数信息和相对位置的三部分结构
        """
        # 1. 获取object信息（包括相对位置）
        if object_dict is None:
            if hasattr(self, 'info') and 'object_dict' in self.info:
                object_dict = self.info["object_dict"]
            else:
                obj_info,_ = self.get_objects_info(self.target_list)
                object_dict = self.enhance_objects_info(obj_info)
        
        # 创建三部分结构的字典
        gt_stats = {
            "agent_info": {},    # 基本信息：名称、特征和状态
            "counting_gt": {},   # 计数统计：总数、类别数等
            "relative_location": {}  # 相对位置关系
        }
        
        # 2. 填充基本信息
        for agent_name in self.target_list:
            agent_type = self.target_agents[agent_name]['agent_type']
            state = self.target_agents[agent_name]['animation']
            
            # 获取feature信息(如果有)
            feature_caption = self.target_agents[agent_name].get('feature_caption', 'None')
            
            refer_name = self.name_mapping[agent_name]
            gt_stats["agent_info"][refer_name] = {
                "agent_type": agent_type,
                "feature": feature_caption,
                "state": state
            }
        
        # 3. 填充计数信息(counting_gt)
        # 计算类别数和每类agent数量
        agent_types = {}
        for agent_name in self.target_list:
            agent_type = self.target_agents[agent_name]['agent_type']
            if agent_type not in agent_types:
                agent_types[agent_type] = []
            agent_types[agent_type].append(agent_name)
        
        # 统计不同状态的agent
        states_count = {}
        for agent_name in self.target_list:
            state = self.target_agents[agent_name]['animation']
            if state != 'None':  # 排除None状态
                if state not in states_count:
                    states_count[state] = 0
                states_count[state] += 1
        
        # 同类别相同状态统计
        type_state_counts = {}
        for agent_name in self.target_list:
            agent_type = self.target_agents[agent_name]['agent_type']
            state = self.target_agents[agent_name]['animation']
            
            if state != 'None':  # 排除None状态
                key = f"{agent_type}_{state}"
                if key not in type_state_counts:
                    type_state_counts[key] = 0
                type_state_counts[key] += 1
        agents_description = {}
        for agent_name_internal in self.target_list:
            refer_name = self.name_mapping.get(agent_name_internal, agent_name_internal) # 获取映射后的名字
            feature_caption = self.target_agents[agent_name_internal].get('feature_caption', 'None')
            agents_description[refer_name] = feature_caption
            
                

        # 汇总计数统计到counting_gt
        gt_stats["counting_gt"] = {
            "total_agents": len(self.target_list),
            "total_agent_types": len(agent_types),
            "agents_per_type": {agent_type: len(agents) for agent_type, agents in agent_types.items()},
            "agents_per_state": states_count,
            "agents_per_type_state": type_state_counts,
            "agents_description": agents_description
        }
        
        # 4. 填充相对位置信息
        gt_stats["relative_location"] = {
            agent_name: info['relative_positions_description']
            for agent_name, info in object_dict.items()
        }
        
        return gt_stats