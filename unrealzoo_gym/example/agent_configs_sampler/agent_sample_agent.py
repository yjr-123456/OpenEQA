import pickle
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
from collections import deque
import json
import os
from openai import OpenAI
from PIL import Image
import io
import base64
import cv2
from dotenv import load_dotenv
import transforms3d
import ast
import re
from agent_configs_sampler.points_sampler import GraphBasedSampler
load_dotenv(override=True)
from .placing_prompt import *

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()

def load_model_config(config_path="model_config.json"):
    """加载模型配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"模型配置文件 {config_path} 不存在，使用默认配置")
        return None
    except json.JSONDecodeError:
        print(f"模型配置文件 {config_path} 格式错误")
        return None

def create_client(model_name="doubao", config_path="model_config.json"):
    """根据模型名称创建OpenAI客户端"""
    config = load_model_config(config_path)
    
    if config is None:
        raise ValueError(f"无法加载模型配置文件 {config_path}")
    
    model_config = config["models"].get(model_name)
    if model_config is None:
        print(f"模型 {model_name} 配置不存在，使用默认模型")
        model_name = config["default_model"]
        model_config = config["models"][model_name]
    key_name = model_config["api_key_env"]
    # 从环境变量获取API密钥
    api_key = os.environ.get(key_name)
    if not api_key:
        raise ValueError(f"环境变量 {key_name} 未设置")

    client = OpenAI(
        base_url=model_config["base_url"],
        api_key=api_key
    )
    
    return client, model_config["model_name"], model_config

# 全局客户端和模型配置
client = None
current_model_name = None
current_model_config = None

def initialize_model(model_name="doubao",model_config_path="model_config.json"):
    """初始化指定的模型"""
    global client, current_model_name, current_model_config
    client, current_model_name, current_model_config = create_client(model_name,model_config_path)
    print(f"已初始化模型: {model_name} ({current_model_name})")


class AgentBasedSampler(GraphBasedSampler):
    def __init__(self, graph_pickle_file, model, config_path="model_config.json"):
        super().__init__(graph_pickle_file)
        self.node_id_list = list(self.node_positions.keys())
        self.node_list = [self.node_positions[node_id] for node_id in self.node_id_list]  # 使用节点ID列表，而不是位置值列表
        self.model = model
        initialize_model(model, f"{config_path}/model_config.json")
    # TO CHECK: 模块化实验
    def run_sampling_experiment(self, env, agent_configs, experiment_config, cam_id=0, cam_count=3, vehicle_zones=None, height=800, **kwargs):
        """
        执行一次完整的、可配置的采样实验。

        Args:
            env: 环境实例。
            agent_configs: Agent配置。
            experiment_config (dict): 实验配置，例如:
                {
                    "use_image": True,  // 是否给VLM看图
                    "prompt_name_object": "default_point_sample", // 对象放置的prompt
                    "prompt_name_camera": "default_camera_sample" // 相机选择的prompt
                }
            cam_id (int): 用于俯视采样的相机ID。
            cam_count (int): 要采样的外部相机数量。
            vehicle_zones (dict): 车辆可放置区域。
            height (int): 俯视相机的高度。
            **kwargs: 其他传递给相机采样函数的参数。
        """
        # 1. 生成采样物体列表
        object_list, all_objects_are_small, has_car = self.sort_objects(agent_configs)
        
        # 2. 预处理车辆区域节点
        vehicle_zone_nodes = self.filter_car_zones(vehicle_zones)
    
        # 3. 采样中心点
        agent_sampling_center_pos, center_node = self.sample_center_point(vehicle_zone_nodes, has_car, all_objects_are_small)
    
        # 4. 设置俯视相机并获取初始视图
        orginal_cam_pose = env.unrealcv.get_cam_location(cam_id) + env.unrealcv.get_cam_rotation(cam_id)
        if agent_sampling_center_pos is not None:
            env.unrealcv.set_cam_location(cam_id, np.append(agent_sampling_center_pos[:2], height))
            env.unrealcv.set_cam_rotation(cam_id, [-90, 0, 0])
        
        obs_bgr = env.unrealcv.read_image(cam_id, 'lit')
        obs_rgb = cv2.cvtColor(obs_bgr, cv2.COLOR_BGR2RGB)
        
        # 更新相机内部状态
        cam_location = env.unrealcv.get_cam_location(cam_id)
        cam_rotation = env.unrealcv.get_cam_rotation(cam_id)
        self.cam_pose = cam_location + cam_rotation
        self.W, self.H = obs_rgb.shape[1], obs_rgb.shape[0]
        self.fov_deg = float(env.unrealcv.get_cam_fov(cam_id))
        self.K = self.get_camera_matrix_unreal(self.W, self.H, self.fov_deg)
        
        # 投影所有可用节点
        img_points, valid_mask, depths = self.project_points_to_image_unreal(self.node_list, self.cam_pose, self.W, self.H, self.fov_deg)
        
        # 5. 准备初始的有效点字典
        all_valid_points_dict = {}
        for i, (node, node_id, valid) in enumerate(zip(self.node_list, self.node_id_list, valid_mask)):
            if valid:
                all_valid_points_dict[tuple(node)] = {'index': i, 'node': node_id}
    
        # 6. 采样主循环：逐个放置物体
        sampled_objects = []
        occupied_areas = []
        pending_objects = []
        
        # 预先确定所有物体的旋转
        for obj_info in object_list:
            agent_type, _, _, name, _, _, _, _ = obj_info
            yaw = self._determine_object_orientation(agent_type)
            pending_objects.append({'name': name, 'rotation': [0, yaw, 0]})

        for i, obj_info in enumerate(object_list):
            agent_type, length, width, name, app_id, animation, feature_caption, type_val = obj_info
            current_obj_details = {
                'name': name, 'length': length, 'width': width, 'agent_type': agent_type,
                'rotation': pending_objects[i]['rotation']
            }
            yaw = current_obj_details['rotation'][1]

            # 生成当前步骤的可视化图像
            result_img = self.visualize_projected_points_unreal_with_next_object(
                obs_rgb, img_points, valid_mask, depths, self.W, self.H, 
                occupied_areas, current_obj_details
            )
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"debug_sampling_step_{i+1}.png", result_img_rgb)

            # !! 核心改动：调用VLM选择放置点，并传入实验配置 !!
            current_node_to_try = self.sample_object_points(
                result_img, name, length, width, all_valid_points_dict, experiment_config
            )
            
            if current_node_to_try is None:
                print(f"警告：为物体 {name} 采样位置失败，跳过此物体。")
                continue

            position = self.node_positions[current_node_to_try]
            
            # 记录已采样的物体信息
            object_dict = {
                'node': current_node_to_try, 'position': position, 'rotation': current_obj_details['rotation'],
                'agent_type': agent_type, 'type': type_val, 'name': name, 'app_id': app_id,
                'animation': animation, 'feature_caption': feature_caption, 'dimensions': (length, width)
            }
            sampled_objects.append(object_dict)
            
            # 更新被占据的区域和有效点
            valid_mask = self._mark_area_occupied(current_node_to_try, occupied_areas, valid_mask, length, width, yaw)
            
            # 更新有效点字典
            all_valid_points_dict = {}
            for j, (node, node_id, valid) in enumerate(zip(self.node_list, self.node_id_list, valid_mask)):
                if valid:
                    all_valid_points_dict[tuple(node)] = {'index': j, 'node': node_id}

        # 7. 采样相机位置
        # 7.1. 采样相机候选位置
        cameras = self._sample_external_cameras(
            objects=sampled_objects, 
            camera_count=cam_count,
            ring_inner_radius_offset=kwargs.get('ring_inner_radius_offset', 200), 
            ring_outer_radius_offset=kwargs.get('ring_outer_radius_offset', 800),
            min_angle_separation_deg=kwargs.get('min_angle_separation_deg', 30),
            min_cam_to_agent_dist=kwargs.get('min_cam_to_agent_dist', 150)
        )
        
        # 7.2. 生成带有相机候选位置的可视化图像
        result_img_with_cams = self.visualize_projected_points_unreal_with_cameras(
            obs_rgb, img_points, valid_mask, depths, self.W, self.H, 
            occupied_areas, cameras
        )
        result_img_rgb = cv2.cvtColor(result_img_with_cams, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"debug_sampling_step_with_cameras.png", result_img_rgb)
        
        # 7.3. !! 核心改动：调用VLM选择最佳相机，并传入实验配置 !!
        camera_id_list = self.sample_camera_points(result_img_with_cams, experiment_config)
        
        selected_cameras = []
        if camera_id_list:
            for cam_info in cameras:
                if cam_info["id"] in camera_id_list:
                    selected_cameras.append(cam_info)

        # 8. 转换为最终的配置格式
        updated_configs, camera_configs = self.format_transform(agent_configs, sampled_objects, selected_cameras)
        center_pos_to_return = agent_sampling_center_pos.tolist() if isinstance(agent_sampling_center_pos, np.ndarray) else agent_sampling_center_pos
        all_distances = [np.linalg.norm(np.array(obj['position']) - agent_sampling_center_pos) for obj in sampled_objects]
        agent_sampling_radius = max(all_distances) + 200 if all_distances else 200
    
        # 9. 恢复原始相机位置
        env.unrealcv.set_cam_location(cam_id, orginal_cam_pose[:3])
        env.unrealcv.set_cam_rotation(cam_id, orginal_cam_pose[3:])
        
        # 10. 返回包含所有信息的字典
        return {
            "env": env,
            'agent_configs': updated_configs,
            'camera_configs': camera_configs,
            'sampling_center': center_pos_to_return,
            'sampling_radius': agent_sampling_radius
        }

    def project_point_to_image(point_3d, rvec, tvec, camera_matrix, dist_coeffs=None):
        """
        Project a 3D point to 2D image coordinates.
        """
        if dist_coeffs is None:
            dist_coeffs = np.zeros((5, 1))  # Assume no distortion
    
        # point_3d should be (1, 3) shape
        point_3d = np.array(point_3d, dtype=np.float32).reshape(1, 1, 3)
        img_points, _ = cv2.projectPoints(point_3d, rvec, tvec, camera_matrix, dist_coeffs)
        return img_points[0][0]  # (x, y)
    
    def pose_to_matrix_unreal(self, cam_pose):
        """直接使用UnrealCV坐标系，不进行坐标系转换"""
        tx, ty, tz, rx, ry, rz = cam_pose
        pitch = np.radians(rx)
        yaw = np.radians(ry)
        roll = np.radians(rz)
        
        # 直接使用UnrealCV的旋转矩阵
        R_mat = transforms3d.euler.euler2mat(-roll, -pitch, yaw, 'sxyz')
        tvec = np.array([tx, ty, tz], dtype=np.float32)
        
        return R_mat, tvec

    def get_camera_matrix_unreal(self, W, H, fov_deg):
        """适配UnrealCV坐标系的相机内参矩阵"""
        fov_rad = np.deg2rad(fov_deg)
        f = (W / 2) / np.tan(fov_rad / 2)
        
        # UnrealCV坐标系下的内参矩阵
        # 可能需要调整主点或其他参数
        K = np.array([[f, 0, W / 2],
                    [0, f, H / 2],
                    [0, 0, 1]])
        return K

    def world_2_image_unreal(self, point_world, cam_pose, K):
        """在UnrealCV坐标系下进行投影"""
        R_mat, tvec = self.pose_to_matrix_unreal(cam_pose)
        
        # 世界坐标转相机坐标（UnrealCV坐标系）
        point_cam = R_mat.T @ (point_world - tvec)
        
        # 在UnrealCV坐标系中，可能需要调整投影方式
        # 试试不同的轴作为深度轴
        
        # 方案1：使用X轴作为深度（UnrealCV中X轴向前）
        if abs(point_cam[0]) > 1e-6:  # 避免除零
            u = K[0,0] * point_cam[1] / point_cam[0] + K[0,2]  # Y/X -> u
            v = K[1,1] * (-point_cam[2]) / point_cam[0] + K[1,2]  # -Z/X -> v (Z向上，图像向下)
            depth = point_cam[0]  # X轴深度
            return int(u), int(v), depth
        else:
            return -1, -1, 0

    def calculate_depth_unreal(self, point_world, cam_pose):
        """在UnrealCV坐标系下计算深度"""
        R_mat, tvec = self.pose_to_matrix_unreal(cam_pose)
        point_cam = R_mat.T @ (point_world - tvec)
        
        # 在UnrealCV中，深度通常是X轴方向（相机前方）
        depth = point_cam[0]
        return depth

    def project_points_to_image_unreal(self, world_points, cam_pose, W, H, fov_deg=90):
        """在UnrealCV坐标系下投影多个点"""
        image_points = []
        depths = []
        valid_mask = []
        self.valid_points = []
        for point_world in world_points:
            u, v, depth = self.world_2_image_unreal(point_world, cam_pose, self.K)
            depths.append(depth)

            if depth > 0 and 5 <= u < W-5 and 5 <= v < H-5:
                valid_mask.append(True)
                self.valid_points.append(point_world)
            else:
                valid_mask.append(False)
                
            image_points.append([u, v])
        
        return np.array(image_points), np.array(valid_mask), np.array(depths)

    def sort_objects(self, agent_configs):
        # 1. 生成采样物体列表
        object_list = []
        car_reference_area = 400 * 200  # 80000
        small_object_area_threshold = car_reference_area * 0.3 # 定义小物体面积阈值 (例如汽车面积的30%)
        all_objects_are_small = True # 初始化标志
        has_car = False # 是否有任何标记为 'car' 类型的物体

        for agent_type, config in agent_configs.items():
            current_agent_is_car_type = (agent_type == 'car')
            if current_agent_is_car_type:
                has_car = True # 标记场景中存在'car'类型的配置

            # 确定尺寸的逻辑
            # (这里的尺寸确定逻辑与您文件中的保持一致)
            size = (100,100) # 默认尺寸
            if agent_type == 'player':
                size = (50, 50)
            elif agent_type == 'car':
                size = (400, 200)
            elif agent_type == 'motorbike':
                size = (180, 60)
            elif agent_type == 'drone':
                size = (100, 100)
            elif agent_type == 'animal':
                # 假设 app_id 列表与 name 列表等长
                # 如果 agent_configs[agent_type]['app_id'] 不存在或为空，需要处理
                # 这里简化，假设 app_id 总是可用的
                # 注意：在循环外确定通用尺寸，在循环内根据app_id细化animal尺寸
                pass # animal 的尺寸将在下面的循环中根据app_id确定

            for i, name in enumerate(config['name']):
                temp_size = size # 默认使用上面确定的尺寸
                if agent_type == 'animal': # 针对animal类型，根据app_id细化尺寸
                    app_id = config['app_id'][i]
                    if app_id == 0: temp_size = (50, 50)
                    elif app_id in [1, 2]: temp_size = (80, 50)
                    elif app_id in [3, 5, 12]: temp_size = (40, 40)
                    elif app_id == 6: temp_size = (300, 150)
                    elif app_id == 9: temp_size = (30, 30)
                    elif app_id in [10, 14]: temp_size = (400, 150)
                    elif app_id in [11, 15, 25, 26]: temp_size = (300, 280)
                    elif app_id in [16, 20, 21, 22]: temp_size = (350, 280)
                    elif app_id == 19: temp_size = (250, 280)
                    elif app_id == 23: temp_size = (400, 280)
                    elif app_id == 27: temp_size = (300, 200)
                    else: temp_size = (50, 50) # animal的默认尺寸
                
                object_area = temp_size[0] * temp_size[1]
                if object_area >= small_object_area_threshold:
                    all_objects_are_small = False
                
                object_list.append((
                    agent_type, temp_size[0], temp_size[1], name, config['app_id'][i],
                    config['animation'][i] if 'animation' in config else 'None',
                    config['feature_caption'][i] if 'feature_caption' in config else '',
                    config['type'][i] if agent_type == 'car' else 'None'
                ))
        object_list.sort(key=lambda x: x[1] * x[2], reverse=True)
        return object_list, all_objects_are_small, has_car

    def filter_car_zones(self, vehicle_zones):
        def point_in_rectangle(point, rectangle_corners):
            import matplotlib.path as mpath
            # 确保 rectangle_corners 是闭合的，或者 mpath.Path 能正确处理
            path = mpath.Path(rectangle_corners)
            return path.contains_point((point[0], point[1]))
    
        vehicle_zone_nodes = {}
        if vehicle_zones: # 确保 vehicle_zones 不是 None
            for zone_type, zones_for_type in vehicle_zones.items():
                current_zone_nodes = []
                for node, pos in self.node_positions.items():
                    for rectangle_corners in zones_for_type: # zones_for_type 是一个矩形列表
                        if point_in_rectangle([pos[0], pos[1]], rectangle_corners):
                            current_zone_nodes.append(node)
                            break # 该节点已在区域内，无需检查此类型的其他区域
                if current_zone_nodes:
                    vehicle_zone_nodes[zone_type] = current_zone_nodes
        return vehicle_zone_nodes

    def sample_center_point(self, vehicle_zone_nodes, has_car, all_objects_are_small):
        if not self.node_id_list:
            print("错误：图中没有节点可供采样。")
            # 返回空配置和None的中心/半径
            return {'agent_configs': {}, 'camera_configs': {}, 'sampling_center': None, 'sampling_radius': None}
        
        center_node_source = "图的随机节点"
        if all_objects_are_small: # all_objects_are_small 和 has_car 在步骤1中计算
            center_node = random.choice(self.node_id_list)
            center_node_source = "图的随机节点 (所有物体较小)"
        elif has_car and vehicle_zone_nodes.get('car'):
            center_node = random.choice(vehicle_zone_nodes['car'])
            center_node_source = "车辆区域的随机节点 (因有车)"
        else:
            center_node = random.choice(self.node_id_list)
        
        print(f"中心点从 {center_node_source} 选择。")
        # agent_sampling_center_node = center_node # 可以保留节点ID如果需要
        agent_sampling_center_pos = np.array(self.node_positions[center_node]) # <--- 这是智能体采样的圆心位置
        return agent_sampling_center_pos, center_node

    def _mark_area_occupied(self, center_node, occupied_areas, valid_mask, length, width, yaw):
        """
        标记被物体占据的节点，更新valid_mask (适配Unreal左手坐标系)
        """
        # 获取中心节点的位置
        center_pos = np.array(self.node_positions[center_node])
        
        # 将角度转换为弧度
        yaw_rad = np.radians(yaw)
        
        # 计算物体四个角的相对偏移（考虑旋转）
        half_length = length / 2.0
        half_width = width / 2.0
        
        # 物体的四个角（相对于中心点，在物体局部坐标系中）
        # 假设物体的"前方"是沿着长度方向（length）
        corners = np.array([
            [-half_length, -half_width],  # 左后
            [half_length, -half_width],   # 右后  
            [half_length, half_width],    # 右前
            [-half_length, half_width]    # 左前
        ])
        
        # 左手坐标系下的旋转矩阵（绕Z轴，yaw角度）
        # 注意：左手系中正yaw是顺时针旋转
        rotation_matrix = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad)],    # 注意这里是+sin
            [np.sin(yaw_rad), np.cos(yaw_rad)]    # 注意这里是-sin
        ])
        
        rotated_corners = corners @ rotation_matrix.T
        
        # 计算物体在世界坐标系中的四个角
        world_corners = rotated_corners + center_pos[:2]
        
        updated_valid_mask = valid_mask.copy()
        
        for i, node_id in enumerate(self.node_id_list): 
            if not valid_mask[i]:
                continue
                
            node_pos = np.array(self.node_positions[node_id])[:2]  
            
            if self._point_in_rotated_rectangle(node_pos, center_pos[:2], length, width, yaw_rad):
                updated_valid_mask[i] = False
                print(f"节点 {node_id} 被物体占据，标记为无效")  
        
        occupied_areas.append({
            'center_node': center_node,
            'center_pos': center_pos,
            'length': length,
            'width': width,
            'yaw': yaw,
            'corners': world_corners
        })
        
        return updated_valid_mask
    
    def _determine_object_orientation(self, obj_type, existing_objects=None, env_context=None):
        """
        完全独立于位置和节点的物体朝向确定函数
        
        Args:
            obj_type: 物体类型 ('car', 'motorbike', 'drone', 'human', 'animal', 'player')
            existing_objects: 已放置的物体列表（可选，用于高级策略）
            env_context: 环境上下文信息（可选）
        
        Returns:
            yaw: 物体的朝向角度（度）
        """
        import math
        import random
        import numpy as np
        
        # 基于物体类型的朝向策略
        if obj_type in ['car', 'motorbike']:
            # 车辆倾向于标准道路方向（0°, 45°, -45°, 135°, -135°, 180°）
            standard_directions = [0, 45, -45, 135, -135, 180]
            base_yaw = random.choice(standard_directions)
            # 添加小的随机偏差让朝向更自然
            return base_yaw + random.uniform(-15, 15)
        
        elif obj_type == 'drone':
            # 无人机可以任意朝向，但偏好开阔方向
            # 可以完全随机，或者偏好某些方向
            return random.uniform(-180, 180)
        
        elif obj_type in ['player']:
            # 人类偏好面向"有趣"的方向
            # 如果有已放置的物体，可能面向它们
            if existing_objects and len(existing_objects) > 0:
                # 30%概率选择面向其他物体的大致方向
                if random.random() < 0.3:
                    # 随机选择一个通用的"社交"朝向
                    social_directions = [0, 45, 90, 135, 180]
                    base_yaw = random.choice(social_directions)
                    return base_yaw + random.uniform(-30, 30)
            
            # 70%概率或无其他物体时：完全随机
            return random.uniform(-180, 180)
        
        elif obj_type == 'animal':
            # 动物的朝向更随机，但可能有一些偏好
            # 例如：偏好避开"危险"方向，或者面向"食物"方向
            
            # 动物可能偏好某些方向（模拟自然行为）
            if random.random() < 0.4:  # 40%概率选择"自然"方向
                natural_directions = [0, 45, -45, 135, -135, 180]  # 主要方向
                base_yaw = random.choice(natural_directions)
                return base_yaw + random.uniform(-60, 60)  # 较大的随机偏差
            else:
                # 60%概率完全随机
                return random.uniform(-180, 180)
        
        else:
            # 未知类型：完全随机朝向
            return random.uniform(-180, 180)
       
    def _point_in_rotated_rectangle(self, point, center, length, width, yaw_rad):
        """
        检查点是否在旋转的矩形内 (适配Unreal左手坐标系)
        """
        # 将点转换到矩形的局部坐标系
        relative_point = point - center
        
        # 应用反向旋转（左手坐标系）
        cos_yaw = np.cos(-yaw_rad)
        sin_yaw = np.sin(-yaw_rad)
        
        # 左手坐标系下的反向旋转
        local_x = relative_point[0] * cos_yaw - relative_point[1] * sin_yaw  # 这里保持不变
        local_y = relative_point[0] * sin_yaw + relative_point[1] * cos_yaw  # 这里保持不变
        
        # 检查是否在矩形边界内（加上安全距离）
        buffer = 50  # 50厘米的安全距离
        half_length = (length + buffer) / 2.0
        half_width = (width + buffer) / 2.0
        
        return abs(local_x) <= half_length and abs(local_y) <= half_width

    def draw_occupied_rectangle(self, result_img, occupied_area, obj_idx, W, H):
        """
        在图像上绘制已占据物体的矩形方框
        
        Args:
            result_img: 要绘制的图像
            occupied_area: 占据区域信息字典
            obj_idx: 物体索引
            W, H: 图像宽高
        """
        try:
            # 获取物体的四个角的世界坐标
            world_corners = occupied_area['corners']  # shape: (4, 2)
            
            # 将四个角投影到图像坐标系
            image_corners = []
            valid_corners = []
            
            for corner in world_corners:
                # 将2D角点扩展为3D（添加Z坐标）
                corner_3d = np.array([corner[0], corner[1], occupied_area['center_pos'][2]])
                
                # 投影到图像坐标
                u, v, depth = self.world_2_image_unreal(corner_3d, self.cam_pose, self.K)
                
                # 检查是否在图像范围内
                if depth > 0 and 0 <= u < W and 0 <= v < H:
                    image_corners.append([u, v])
                    valid_corners.append(True)
                else:
                    image_corners.append([u, v])  # 即使无效也保存坐标
                    valid_corners.append(False)
            
            # 如果至少有2个角在图像内，绘制矩形
            valid_count = sum(valid_corners)
            if valid_count >= 2:
                # 转换为numpy数组并取整
                corners_array = np.array(image_corners, dtype=np.int32)
                
                # 绘制矩形轮廓
                cv2.polylines(result_img, [corners_array], isClosed=True, 
                            color=(0, 0, 255), thickness=2)  # 红色矩形
                
                # 在矩形中心添加物体标签
                center_u = int(np.mean(corners_array[:, 0]))
                center_v = int(np.mean(corners_array[:, 1]))
                
                # 确保标签在图像范围内
                if 0 <= center_u < W and 0 <= center_v < H:
                    # 添加半透明背景
                    label_text = f"Obj{obj_idx}"
                    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # 绘制文本背景
                    cv2.rectangle(result_img, 
                                (center_u - text_size[0]//2 - 2, center_v - text_size[1] - 2),
                                (center_u + text_size[0]//2 + 2, center_v + 2),
                                (0, 0, 255), -1)  # 红色背景
                    
                    # 绘制文本
                    cv2.putText(result_img, label_text, 
                            (center_u - text_size[0]//2, center_v), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # 白色文字
                    
        except Exception as e:
            print(f"绘制物体{obj_idx}的矩形时出错: {e}")

    def convert_depth_to_8bit(self,depth_array, method='linear',min_val=None,max_val=None,
                              invert=False,gamma=1.0,log_scale=False,clip_percentile=None):
        """
        将深度图转换为8位格式
        
        参数:
            depth_array: 原始深度图数组
            method: 转换方法，支持 'linear'(线性), 'inverse'(倒数)
            min_val, max_val: 指定归一化范围，默认为数组的最小值和最大值
            invert: 是否反转深度值（近白远黑或近黑远白）
            gamma: gamma校正值，默认1.0（不校正）
            log_scale: 是否使用对数缩放
            clip_percentile: 百分比截断，例如[2, 98]截断最亮和最暗的2%
            
        返回:
            depth_8bit: 8位深度图 (0-255)
            metadata: 元数据信息，用于后续恢复
        """
        import numpy as np
        
        # 复制数组防止修改原数据
        depth = depth_array.copy()
        
        # 保存原始信息
        original_min = float(depth.min())
        original_max = float(depth.max())
        
        # 应用倒数处理
        if method == 'inverse':
            # 避免除零
            depth = 1.0 / (depth + 1e-8)
        
        # 过滤无效值
        valid_mask = ~np.isnan(depth) & ~np.isinf(depth) & (depth > 0)
        valid_min = depth[valid_mask].min() if valid_mask.any() else 0
        valid_max = depth[valid_mask].max() if valid_mask.any() else 1
        
        # 使用指定的范围或有效数据范围
        min_val = valid_min if min_val is None else min_val
        max_val = valid_max if max_val is None else max_val
        
        # 应用百分比截断
        if clip_percentile is not None:
            low, high = clip_percentile
            if valid_mask.any():
                p_low = np.percentile(depth[valid_mask], low)
                p_high = np.percentile(depth[valid_mask], high)
                depth = np.clip(depth, p_low, p_high)
                min_val = p_low
                max_val = p_high
        
        # 应用对数缩放
        if log_scale and valid_mask.any():
            depth[valid_mask] = np.log1p(depth[valid_mask] - min_val + 1e-8)
            max_log = np.log1p(max_val - min_val + 1e-8)
            depth = depth / max_log
        else:
            # 线性归一化
            depth_range = max_val - min_val
            if depth_range > 0:
                depth = (depth - min_val) / depth_range
            else:
                depth = np.zeros_like(depth)
        
        # 应用gamma校正
        if gamma != 1.0:
            depth = np.power(depth, gamma)
        
        # 反转（如果需要）
        if invert:
            depth = 1.0 - depth
        
        # 转换为8位
        depth_8bit = (depth * 255).astype(np.uint8)
        
        # 创建元数据
        metadata = {
            'original_min': original_min,
            'original_max': original_max,
            'process_min': float(min_val),
            'process_max': float(max_val),
            'method': method,
            'gamma': gamma,
            'log_scale': log_scale,
            'invert': invert,
            'clip_percentile': clip_percentile
        }
        
        return depth_8bit, metadata

    def sample_agent_positions(self, env, agent_configs, cam_id=0,
                                    cam_count=3, vehicle_zones=None, height=800,**kwargs): # 添加 **kwargs
        """
        更高内聚度采样：先定中心点，再在圆形区域内采样所有物体。
        现在会将相机采样相关的kwargs传递给_sample_external_cameras。
        """
        # 1. 生成采样物体列表
        object_list, all_objects_are_small, has_car = self.sort_objects(agent_configs)
        # 2. 预处理车辆区域节点
        vehicle_zone_nodes = self.filter_car_zones(vehicle_zones)
    
        # 3. 采样中心点
        agent_sampling_center_pos, center_node = self.sample_center_point(vehicle_zone_nodes, has_car, all_objects_are_small)
    
        # 4. 设置相机
        orginal_cam_pose = env.unrealcv.get_cam_location(cam_id) + env.unrealcv.get_cam_rotation(cam_id)
        if agent_sampling_center_pos is not None:
            env.unrealcv.set_cam_location(cam_id, np.append(agent_sampling_center_pos[:2],height))
            env.unrealcv.set_cam_rotation(cam_id, [-90, 0, 0])
        obs_bgr = env.unrealcv.read_image(cam_id,'lit')
        obs_rgb = cv2.cvtColor(obs_bgr, cv2.COLOR_BGR2RGB)
        obs_depth = env.unrealcv.get_depth(cam_id)
        depth_8bit, _ = self.convert_depth_to_8bit(obs_depth, method='inverse')
        cam_location = env.unrealcv.get_cam_location(cam_id)
        cam_rotation = env.unrealcv.get_cam_rotation(cam_id)
        self.cam_pose = cam_location + cam_rotation
        self.W, self.H = obs_rgb.shape[1], obs_rgb.shape[0]
        self.fov_deg = float(env.unrealcv.get_cam_fov(cam_id))
        self.K = self.get_camera_matrix_unreal(self.W, self.H, self.fov_deg)
        img_points,valid_mask, depths = self.project_points_to_image_unreal(self.node_list,self.cam_pose,self.W,self.H,self.fov_deg)
        # 5.图像内所有点
        # all_valid_points = [dict(self.node_positions[node], node=node) for node, valid in zip(self.node_list, valid_mask) if valid]
        all_valid_points_dict = {}
        cnt_index = 0
        for node,node_id,valid in zip(self.node_list, self.node_id_list,valid_mask):
            if valid:
                pos = tuple(node)
                all_valid_points_dict[pos] = {
                    'index': cnt_index,
                    'node': node_id
                    }
            cnt_index += 1
    
        # 6. 采样主循环
        sampled_objects = []
        occupied_areas = [] # (node, length, width, rotation)
        pending_objects = []
        for obj_info in object_list:
            agent_type, length, width, name, app_id, animation, feature_caption, type = obj_info
            # 预先确定旋转角度
            yaw = self._determine_object_orientation(agent_type)
            rotation = [0, yaw, 0]
            
            pending_objects.append({
                'name': name,
                'length': length,
                'width': width,
                'agent_type': agent_type,
                'rotation': rotation  # 添加rotation信息
            })

        for i, obj_info in enumerate(object_list):
            agent_type, length, width, name, app_id, animation, feature_caption, type = obj_info
            current_obj = pending_objects[i]
            yaw = current_obj['rotation'][1]
            rotation = current_obj['rotation']
            # 使用扩展版可视化（传入当前待放置物体列表）
            # current_pending = pending_objects[i:]  # 剩余待放置的物体
            # result_img = self.visualize_projected_points_unreal_extended(
            #     obs_rgb, img_points, valid_mask, depths, self.W, self.H, 
            #     occupied_areas, current_pending)
            # cv2.imwrite(f"debug_sampling_step_{i+1}.png", result_img)
            # result_img= self.visualize_projected_points_unreal(obs_rgb, img_points,valid_mask, depths, self.W, self.H, occupied_areas)
            result_img = self.visualize_projected_points_unreal_with_next_object(
                    obs_rgb, img_points, valid_mask, depths, self.W, self.H, 
                    occupied_areas, current_obj)
            result_img_rgb= cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"debug_sampling_step_{i+1}.png", result_img_rgb)
            current_node_to_try = self.sample_object_points(result_img, name, length, width, all_valid_points_dict)
            position = self.node_positions[current_node_to_try]
            object_dict = {
                        'node': current_node_to_try,
                        'position': position,
                        'rotation': rotation,
                        'agent_type': agent_type,
                        'type': type, 
                        'name': name,
                        'app_id': app_id,
                        'animation': animation,
                        'feature_caption': feature_caption,
                        'dimensions': (length, width)
                    }
            sampled_objects.append(object_dict)
            valid_mask= self._mark_area_occupied(current_node_to_try, occupied_areas, valid_mask,length, width, yaw)
            # result_img = self.visualize_projected_points_unreal(obs_rgb, img_points, valid_mask, depths, self.W, self.H, occupied_areas)
            # update all_valid_points_dict
            all_valid_points_dict = {}
            cnt_index = 0
            for node,node_id,valid in zip(self.node_list, self.node_id_list,valid_mask):
                if valid:
                    pos = tuple(node)
                    all_valid_points_dict[pos] = {
                        'index': cnt_index,
                        'node': node_id
                        }
                cnt_index += 1
        
        # 7. 计算采样的半径
        all_distances = [np.linalg.norm(np.array(obj['position']) - agent_sampling_center_pos) for obj in sampled_objects]
        agent_sampling_radius = max(all_distances)+200 if all_distances else 200

        # 7. 采样相机 (将kwargs传递下去)
        cameras = self._sample_external_cameras(
            objects=sampled_objects, 
            camera_count=cam_count,
            # 传递新方法所需的参数，或依赖其默认值
            ring_inner_radius_offset=kwargs.get('ring_inner_radius_offset', 200), 
            ring_outer_radius_offset=kwargs.get('ring_outer_radius_offset', 800),
            min_angle_separation_deg=kwargs.get('min_angle_separation_deg', 30),
            min_cam_to_agent_dist=kwargs.get('min_cam_to_agent_dist', 150) # 仍然重要，用于定义圆环内边界
            # fov_deg 和 desired_closeness_factor 在这个新方法中作用不大
        )
        env.unrealcv.set_cam_location(cam_id, np.append(agent_sampling_center_pos[:2], height+200))
        # env.unrealcv.set_cam_fov(cam_id, 120)
        cam_location = env.unrealcv.get_cam_location(cam_id)
        cam_rotation = env.unrealcv.get_cam_rotation(cam_id)
        self.cam_pose = cam_location + cam_rotation
        self.W, self.H = obs_rgb.shape[1], obs_rgb.shape[0]
        self.fov_deg = float(env.unrealcv.get_cam_fov(cam_id))
        self.K = self.get_camera_matrix_unreal(self.W, self.H, self.fov_deg)
        result_img = self.visualize_projected_points_unreal_with_cameras(
        obs_rgb, img_points, valid_mask, depths, self.W, self.H, 
        occupied_areas, cameras)
        camera_id_list = self.sample_camera_points(result_img)
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"debug_sampling_step_with_cameras.png", result_img_rgb)
        selected_cameras = []
        for i, cam_info in enumerate(cameras):
            if cam_info["id"] in camera_id_list:
                selected_cameras.append(cam_info)

        # 8. 转换为配置格式
        updated_configs, camera_configs = self.format_transform(agent_configs, sampled_objects, selected_cameras)
        center_pos_to_return = agent_sampling_center_pos.tolist() if isinstance(agent_sampling_center_pos, np.ndarray) else agent_sampling_center_pos
    
        # 9. 恢复相机位置
        env.unrealcv.set_cam_location(cam_id, orginal_cam_pose[:3])
        env.unrealcv.set_cam_rotation(cam_id, orginal_cam_pose[3:])
        # env.unrealcv.set_cam_fov(cam_id, 90)
        # 返回包含所有信息的字典
        return {
            "env": env,
            'agent_configs': updated_configs,
            'camera_configs': camera_configs,
            'sampling_center': center_pos_to_return,
            'sampling_radius': agent_sampling_radius
        }

    def format_transform(self, agent_configs, sampled_objects, cameras):
        updated_configs = {}
        for agent_type_key in agent_configs.keys(): 
            updated_configs[agent_type_key] = {
                'name': [], 'app_id': [], 'animation': [], 'feature_caption':[], 'start_pos': [], 'type': []
            }

        for obj in sampled_objects:
            agent_type = obj['agent_type']
            if agent_type not in updated_configs: 
                updated_configs[agent_type] = {'name': [], 'app_id': [], 'animation': [], 'feature_caption':[], 'start_pos': [], 'type': []}
            
            updated_configs[agent_type]['name'].append(obj['name'])
            updated_configs[agent_type]['app_id'].append(obj['app_id'])
            updated_configs[agent_type]['animation'].append(obj['animation'])
            updated_configs[agent_type]['feature_caption'].append(obj['feature_caption'])
            updated_configs[agent_type]['type'].append(obj['type']) # 添加类型字段
            pos = obj['position']
            rot = obj['rotation']
            start_pos = [pos[0], pos[1], pos[2], rot[0], rot[1], rot[2]]
            updated_configs[agent_type]['start_pos'].append(start_pos)
    
        camera_configs = {}
        if cameras:
            camera_names = []
            camera_app_ids = []
            camera_animations = [] 
            camera_feature_captions = [] 
            camera_positions = []
            for i, cam in enumerate(cameras):
                camera_names.append(f"camera_{i}")
                camera_app_ids.append(0) 
                camera_animations.append("None")
                camera_feature_captions.append("External camera view") 
                pos = cam["position"]
                rot = cam["rotation"]
                camera_positions.append([pos[0], pos[1], pos[2], rot[0], rot[1], rot[2]])
            camera_configs["camera"] = {
                "name": camera_names,
                "app_id": camera_app_ids,
                "animation": camera_animations,
                "feature_caption": camera_feature_captions, 
                "start_pos": camera_positions
            }
        return updated_configs, camera_configs

    def sample_object_points(self, obs, obj_name, length, width, valid_points, exp_config=None):
        positions = []
        node_indices = []
        node_names = []
        max_tries = 5
        key_list = random.sample(list(valid_points.keys()), k=len(valid_points)) 
        for key in key_list:
            pos = key
            info = valid_points[key]
            positions.append(pos)
            node_indices.append(info['index'])
            node_names.append(info['node'])
        # for pos, info in valid_points.items():
        #     positions.append(pos)
        #     node_indices.append(info['index'])
        #     node_names.append(info['node'])
        valid_points_str = "\n".join([f"node{i}: {pos}  " for i, pos in zip(node_indices, positions)])
        use_img = True
        # if exp_config:
        #     prompt_name = exp_config.get("prompt_name", "default_point_sample")
        #     prompts = PROMPT_TEMPLATES[prompt_name]
        #     sys_prompt_point_sample = prompts["system"]
        #     usr_prompt_point_sample = prompts["user"]
        #     use_img = exp_config.get("use_image", True)
        usr_prompt_local = f"""
        points information: {valid_points_str},\n
        Please select the most suitable point from the valid points for placing the object based on its size.
        Remember to consider the object's dimensions to avoid collisions with other objects.
        """
        sys_prompt_point_sample, usr_prompt_point_sample = load_prompt("sample_point_prompt_cot")
        usr_prompt = f"{usr_prompt_point_sample}\n\n\n{usr_prompt_local}"
        encode_obs = [self.encode_image_array(obs)]
        for attempt in range(max_tries):
            try:
                response = self.call_api_vlm(sys_prompt_point_sample, usr_prompt, encode_obs, use_img=use_img)
                node_id_match = re.search(r'<a>(.*?)</a>', response, re.DOTALL)
                node_position_match = re.search(r'<b>(.*?)</b>', response, re.DOTALL)
                analyze_info_match = re.search(r'<c>(.*?)</c>', response, re.DOTALL)
                if node_id_match and node_position_match:
                    node_id = int(node_id_match.group(1).strip())
                    position = tuple(ast.literal_eval(node_position_match.group(1).strip()))
                    if analyze_info_match:
                        analyze_info = analyze_info_match.group(1).strip()
                        print(f"[Analysis]: {analyze_info}")
                    assert valid_points[position]['index'] == node_id, f"返回的节点ID和位置不匹配,返回节点：{node_id},位置：{position},对应节点：{valid_points[position]['index']}"
                    return valid_points[position]['node']
                else:
                    print(f"尝试 {attempt+1}/{max_tries}：未能从响应中提取节点信息，重新尝试...")
            except Exception as e:
                print(f"尝试 {attempt+1}/{max_tries}：发生错误 {e}，重新尝试...")
        return None

    def sample_camera_points(self, obs):
        max_tries = 5
        encode_obs = [self.encode_image_array(obs)]
        sys_prompt_camera, usr_prompt_camera = load_prompt("sample_camera_point_cot")
        for attempt in range(max_tries):
            try:
                response = self.call_api_vlm(sys_prompt_camera, usr_prompt_camera, encode_obs)
                node_id_match = re.search(r'<a>(.*?)</a>', response, re.DOTALL)
                # node_position_match = re.search(r'<b>(.*?)</b>', response, re.DOTALL)
                if node_id_match:
                    node_id_list = list(ast.literal_eval(node_id_match.group(1).strip()))
                    return node_id_list
                else:
                    print(f"尝试 {attempt+1}/{max_tries}：未能从响应中提取节点信息，重新尝试...")
            except Exception as e:
                print(f"尝试 {attempt+1}/{max_tries}：发生错误 {e}，重新尝试...")
        return None

    def visualize_projected_points_unreal(self, obs_rgb, image_points, valid_mask, depths, W, H, occupied_areas= None):
        """
        在图像上可视化投影的点 (UnrealCV坐标系版本)
        """
        # image_points, valid_mask, depths = self.project_points_to_image_unreal(world_points, cam_pose, W, H, fov_deg)

        result_img = obs_rgb.copy()
        
        for i, (point_img, is_valid, depth) in enumerate(zip(image_points, valid_mask, depths)):
            u, v = point_img
            
            if is_valid:
                # 在图像内的点用绿色标记
                cv2.circle(result_img, (int(u), int(v)), 5, (0, 255, 0), -1)
                
                # 智能调整文本位置
                text_x, text_y = self.get_smart_text_position(u, v, W, H, i)
                cv2.putText(result_img, f"{i}", (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if occupied_areas:
            for obj_idx, occupied_area in enumerate(occupied_areas):
                self.draw_occupied_rectangle(result_img, occupied_area, obj_idx, W, H)
        return result_img

    def get_smart_text_position(self, u, v, W, H, text_content):
        """
        根据点的位置智能调整文本位置，避免超出图像边界
        
        Args:
            u, v: 点的像素坐标
            W, H: 图像宽度和高度
            text_content: 文本内容（用于估算文本大小）
        
        Returns:
            text_x, text_y: 调整后的文本位置
        """
        # 估算文本大小（粗略估计）
        text_width = len(str(text_content)) * 10  # 每个字符大约10像素
        text_height = 15  # 文本高度大约15像素
        
        # 默认偏移量
        offset_x = 5
        offset_y = -5
        
        # 根据点在图像中的位置调整文本位置
        
        # 右边界检查：如果点靠近右边界，文本放在点的左侧
        if u + offset_x + text_width > W:
            offset_x = -(text_width + 5)
        
        # 左边界检查：如果点靠近左边界，文本放在点的右侧
        if u + offset_x < 0:
            offset_x = 5
        
        # 上边界检查：如果点靠近上边界，文本放在点的下方
        if v + offset_y - text_height < 0:
            offset_y = text_height + 5
        
        # 下边界检查：如果点靠近下边界，文本放在点的上方
        if v + offset_y > H:
            offset_y = -5
        
        # 计算最终文本位置
        text_x = int(u + offset_x)
        text_y = int(v + offset_y)
        
        # 最终边界限制（确保文本完全在图像内）
        text_x = max(0, min(W - text_width, text_x))
        text_y = max(text_height, min(H, text_y))
        
        return text_x, text_y

    def call_api_vlm(self, sys_prompt, usr_prompt, base64_image_list=[], use_img = True):
        """
        Call the vLM API with the given prompt.
        """
        messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": usr_prompt,
                    }
                ]
                }
        ]
        if use_img and base64_image_list:
            for base64_image in base64_image_list:
                    messages.append({"role": "user", "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            }
                        }]
                    })
        # Assuming OpenAI API is set up correctly
        response = client.chat.completions.create(
            model=current_model_name,
            max_tokens=9000,
            messages=messages
        )
        respon=  response.choices[0].message.content.strip()
        print(f"[VLM RESPONSE] {respon}")
        return respon

    def encode_image_array(self, image_array):
        # Convert the image array to a PIL Image object
        image = Image.fromarray(np.uint8(image_array))

        # Save the PIL Image object to a bytes buffer
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")

        # Encode the bytes buffer to Base64
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return img_str
    
    def visualize_projected_points_unreal_extended(self, obs_rgb, image_points, valid_mask, depths, W, H, 
                                                occupied_areas=None, pending_objects=None, height=800):
        """
        扩展版可视化：在图像右侧和下方添加待放置物体信息和尺度标尺
        
        Args:
            obs_rgb: 原始图像
            image_points: 投影点
            valid_mask: 有效性掩码
            depths: 深度信息
            W, H: 原图像宽高
            occupied_areas: 已占据区域
            pending_objects: 待放置物体列表，包含rotation信息
                            [{'name': str, 'length': int, 'width': int, 'agent_type': str, 'rotation': [rx, ry, rz]}, ...]
        """
        import cv2
        import numpy as np
        import math
        
        # 扩展画布尺寸
        info_panel_width = 350  # 增加宽度以容纳旋转信息
        scale_panel_height = 100
        extended_W = W + info_panel_width
        extended_H = H + scale_panel_height
        
        # 创建扩展的画布
        extended_img = np.ones((extended_H, extended_W, 3), dtype=np.uint8) * 240
        
        # 将原图像放在左上角
        extended_img[0:H, 0:W] = obs_rgb.copy()
        
        # 绘制原有的投影点和占据区域
        for i, (point_img, is_valid, depth) in enumerate(zip(image_points, valid_mask, depths)):
            u, v = point_img
            
            if is_valid:
                cv2.circle(extended_img, (int(u), int(v)), 5, (0, 255, 0), -1)
                text_x, text_y = self.get_smart_text_position(u, v, W, H, i)
                cv2.putText(extended_img, f"{i}", (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 绘制已占据区域
        if occupied_areas:
            for obj_idx, occupied_area in enumerate(occupied_areas):
                self.draw_occupied_rectangle(extended_img, occupied_area, obj_idx, W, H)
        
        # 绘制分割线
        cv2.line(extended_img, (W, 0), (W, H), (100, 100, 100), 2)
        cv2.line(extended_img, (0, H), (W, H), (100, 100, 100), 2)
        
        # 绘制右侧信息面板（包含旋转信息）
        self.draw_object_info_panel_with_rotation(extended_img, W, H, info_panel_width, pending_objects, occupied_areas)
        
        # 绘制下方尺度面板
        self.draw_scale_panel(extended_img, W, H, scale_panel_height, extended_W, height)
        
        return extended_img

    def draw_scale_panel(self, img, start_x, start_y, panel_height, panel_width, height=800):
        """
        在下方绘制尺度标尺面板
        """
        import cv2
        import numpy as np
        
        panel_start_y = start_y + 10
        scale_y = panel_start_y + panel_height // 2
        
        # 标题
        cv2.putText(img, "Scale Reference", (20, panel_start_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # 计算100cm在图像中的像素长度（基于相机参数和平均深度）
        if hasattr(self, 'cam_pose') and hasattr(self, 'K'):
            # 估算100cm的像素长度
            estimated_depth = height  # 假设相机高度800cm
            pixel_per_cm = self.K[0, 0] / estimated_depth  # 粗略估算
            scale_100cm_pixels = int(100 * pixel_per_cm)
            
            # 限制标尺长度不超过面板宽度的70%
            max_scale_length = int(panel_width * 0.7)
            if scale_100cm_pixels > max_scale_length:
                scale_100cm_pixels = max_scale_length
                actual_cm = int(100 * max_scale_length / (100 * pixel_per_cm))
                scale_text = f"{actual_cm}cm (adjusted)"
            else:
                scale_text = "100cm"
        else:
            # 如果没有相机参数，使用固定长度
            scale_100cm_pixels = 100
            scale_text = "100cm (estimated)"
        
        # 绘制标尺
        scale_start_x = 20
        scale_end_x = scale_start_x + scale_100cm_pixels
        
        # 主标尺线
        cv2.line(img, (scale_start_x, scale_y), (scale_end_x, scale_y), (0, 0, 0), 3)
        
        # 端点竖线
        cv2.line(img, (scale_start_x, scale_y - 10), (scale_start_x, scale_y + 10), (0, 0, 0), 2)
        cv2.line(img, (scale_end_x, scale_y - 10), (scale_end_x, scale_y + 10), (0, 0, 0), 2)
        
        # 标尺刻度（每10cm一个小刻度）
        num_ticks = min(10, scale_100cm_pixels // 10)  # 最多10个刻度
        if num_ticks > 1:
            for i in range(1, num_ticks):
                tick_x = scale_start_x + (scale_100cm_pixels * i // num_ticks)
                cv2.line(img, (tick_x, scale_y - 5), (tick_x, scale_y + 5), (0, 0, 0), 1)
        
        # 标尺文字
        text_x = scale_start_x + scale_100cm_pixels // 2 - 30
        cv2.putText(img, scale_text, (text_x, scale_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 添加相机信息
        if hasattr(self, 'cam_pose') and hasattr(self, 'fov_deg'):
            cam_info = f"Camera: FOV={self.fov_deg:.1f}°, Height={self.cam_pose[2]:.0f}cm"
            cv2.putText(img, cam_info, (scale_end_x + 50, scale_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    def draw_object_info_panel_with_rotation(self, img, start_x, start_y, panel_width, pending_objects, occupied_areas):
        """
        在右侧绘制包含旋转信息的物体信息面板
        """
        import cv2
        import math
        import numpy as np
        
        panel_start_x = start_x + 10
        current_y = 20
        
        # 标题
        cv2.putText(img, "Object Status", (panel_start_x, current_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        current_y += 30
        
        # 已放置物体
        if occupied_areas:
            cv2.putText(img, f"Placed ({len(occupied_areas)}):", (panel_start_x, current_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1)
            current_y += 20
            
            for i, area in enumerate(occupied_areas):
                obj_name = area.get('name', f'Obj{i}')
                length = area.get('length', 0)
                width = area.get('width', 0)
                yaw = area.get('yaw', 0)
                
                # 绘制旋转的物体图标 - 修复函数名
                icon_center_x = panel_start_x + 20
                icon_center_y = current_y
                
                # 计算图标像素尺寸
                icon_length = length * 0.1  # 使用固定缩放
                icon_width = width * 0.1
                
                self.draw_rotated_object_icon_real_size(img, icon_center_x, icon_center_y, 
                                                       icon_length, icon_width, yaw, 
                                                       color=(0, 0, 255))  # ✅ 使用正确的方法名
                
                # 物体文本信息（包含旋转角度）
                text = f"{obj_name} {length}x{width}cm"
                rotation_text = f"Yaw: {yaw:.1f}°"
                
                cv2.putText(img, text, (panel_start_x + 50, current_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                cv2.putText(img, rotation_text, (panel_start_x + 50, current_y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
                
                current_y += 35
        
        current_y += 10
        
        # 待放置物体
        if pending_objects:
            cv2.putText(img, f"Pending ({len(pending_objects)}):", (panel_start_x, current_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 0), 1)
            current_y += 20
            
            for obj in pending_objects:
                name = obj.get('name', 'Unknown')
                length = obj.get('length', 0)
                width = obj.get('width', 0)
                agent_type = obj.get('agent_type', '')
                rotation = obj.get('rotation', [0, 0, 0])
                yaw = rotation[1] if len(rotation) > 1 else 0
                
                # 根据物体类型选择颜色
                type_colors = {
                    'car': (255, 100, 100),
                    'motorbike': (100, 255, 100),
                    'player': (100, 100, 255),
                    'animal': (255, 255, 100),
                    'drone': (255, 100, 255)
                }
                color = type_colors.get(agent_type, (150, 150, 150))
                
                # 绘制旋转的物体图标 - 修复函数名
                icon_center_x = panel_start_x + 20
                icon_center_y = current_y
                
                # 计算图标像素尺寸
                icon_length = length * 0.1  # 使用固定缩放
                icon_width = width * 0.1
                
                self.draw_rotated_object_icon_real_size(img, icon_center_x, icon_center_y, 
                                                       icon_length, icon_width, yaw, 
                                                       color=color)  
                
                # 物体文本信息（包含预设旋转角度）
                text = f"{name} {length}x{width}cm"
                rotation_text = f"Yaw: {yaw:.1f}° (planned)"
                
                cv2.putText(img, text, (panel_start_x + 50, current_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                cv2.putText(img, rotation_text, (panel_start_x + 50, current_y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
                
                current_y += 35
    
    def calculate_real_world_scale(self, target_depth=None):
        """
        基于相机参数计算真实世界尺寸到像素的缩放比例
        
        Args:
            target_depth: 目标深度（cm），如果为None则使用相机高度
    
        Returns:
            scale: 1cm对应多少像素
        """
        if not hasattr(self, 'cam_pose') or not hasattr(self, 'K'):
            # 如果没有相机参数，返回默认缩放
            return 0.1
        
        # 使用相机高度作为参考深度
        if target_depth is None:
            target_depth = self.cam_pose[2]  # 相机高度
    
        # 焦距（像素）
        focal_length = self.K[0, 0]
        
        # 在指定深度下，1cm对应多少像素
        # 公式：pixel_size = focal_length * real_size / depth
        scale = focal_length / target_depth
        
        return scale

    def visualize_projected_points_unreal_with_next_object(self, obs_rgb, image_points, valid_mask, depths, W, H, 
                                                     occupied_areas=None, next_object=None):
        """
        在投影图右侧显示下一个待放置物体的1:1真实尺寸预览
        
        Args:
            obs_rgb: 原始图像
            image_points: 投影点
            valid_mask: 有效性掩码
            depths: 深度信息
            W, H: 原图像宽高
            occupied_areas: 已占据区域
            next_object: 下一个待放置的物体信息
                        {'name': str, 'length': int, 'width': int, 'agent_type': str, 'rotation': [rx, ry, rz]}
        """
        import cv2
        import numpy as np
        import math
        if not next_object:
            result_img = obs_rgb.copy()
            # 绘制原有的投影点
            for i, (point_img, is_valid, depth) in enumerate(zip(image_points, valid_mask, depths)):
                u, v = point_img
                if is_valid:
                    cv2.circle(result_img, (int(u), int(v)), 5, (0, 255, 0), -1)
                    text_x, text_y = self.get_smart_text_position(u, v, W, H, i)
                    cv2.putText(result_img, f"{i}", (text_x, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # 绘制已占据区域
            if occupied_areas:
                for obj_idx, occupied_area in enumerate(occupied_areas):
                    self.draw_occupied_rectangle(result_img, occupied_area, obj_idx, W, H)
            return result_img


        # 计算真实世界缩放比例
        real_scale = self.calculate_real_world_scale()
        
        # 计算下一个物体的真实像素尺寸
        preview_width = 200  # 预览区域的最小宽度
        if next_object:
            obj_length_px = int(next_object['length'] * real_scale)
            obj_width_px = int(next_object['width'] * real_scale)
            # 确保预览区域足够容纳物体并有边距
            preview_width = max(preview_width, obj_length_px + 100, obj_width_px + 100)
        
        # 扩展画布
        extended_W = W + preview_width
        extended_H = H
    
        # 创建扩展的画布
        extended_img = np.ones((extended_H, extended_W, 3), dtype=np.uint8) * 240
        
        # 将原图像放在左侧
        extended_img[0:H, 0:W] = obs_rgb.copy()
        
        # 绘制原有的投影点和占据区域
        for i, (point_img, is_valid, depth) in enumerate(zip(image_points, valid_mask, depths)):
            u, v = point_img
            
            if is_valid:
                cv2.circle(extended_img, (int(u), int(v)), 5, (0, 255, 0), -1)
                text_x, text_y = self.get_smart_text_position(u, v, W, H, i)
                cv2.putText(extended_img, f"{i}", (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
        # 绘制已占据区域
        if occupied_areas:
            for obj_idx, occupied_area in enumerate(occupied_areas):
                self.draw_occupied_rectangle(extended_img, occupied_area, obj_idx, W, H)
        
        # 绘制分割线
        cv2.line(extended_img, (W, 0), (W, H), (100, 100, 100), 3)
    
        # 绘制右侧预览区域
        if next_object:
            self.draw_next_object_preview(extended_img, W, preview_width, H, next_object, real_scale)
        
        return extended_img

    def visualize_projected_points_unreal_with_cameras(self, obs_rgb, image_points, valid_mask, depths, W, H, 
                                                    occupied_areas=None, camera_candidates=None):
        """
        在投影图上显示放置的物体和相机候选位置（无侧栏）
        
        Args:
            obs_rgb: 原始图像
            image_points: 投影点
            valid_mask: 有效性掩码
            depths: 深度信息
            W, H: 原图像宽高
            occupied_areas: 已占据区域
            camera_candidates: 相机候选位置列表
                            [{'position': [x, y, z], 'rotation': [rx, ry, rz], 'id': int}, ...]
        """
        import cv2
        import numpy as np
        import math
        
        # 直接使用原图像，不扩展画布
        result_img = obs_rgb.copy()
        
        # 绘制投影点
        # for i, (point_img, is_valid, depth) in enumerate(zip(image_points, valid_mask, depths)):
        #     u, v = point_img
            
        #     if is_valid:
        #         cv2.circle(result_img, (int(u), int(v)), 5, (0, 255, 0), -1)
        #         text_x, text_y = self.get_smart_text_position(u, v, W, H, i)
        #         cv2.putText(result_img, f"{i}", (text_x, text_y), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 绘制已占据区域（红色矩形）
        if occupied_areas:
            for obj_idx, occupied_area in enumerate(occupied_areas):
                self.draw_occupied_rectangle(result_img, occupied_area, obj_idx, W, H)
        
        # 绘制相机候选位置
        if camera_candidates:
            self.draw_camera_positions_on_main_image(result_img, camera_candidates, W, H)
        
        return result_img

    def draw_camera_positions_on_main_image(self, img, camera_candidates, W, H):
        """
        在主图像上绘制相机候选位置
        """
        import cv2
        import numpy as np
        import math
        
        for i, cam in enumerate(camera_candidates):
            # 将相机3D位置投影到图像上
            cam_pos_3d = np.array(cam['position'])
            u, v, depth = self.world_2_image_unreal(cam_pos_3d, self.cam_pose, self.K)
            
            # 如果相机位置在图像范围内，绘制相机图标
            if depth > 0 and 0 <= u < W and 0 <= v < H:
                # 绘制相机图标（蓝色圆圈）
                cv2.circle(img, (int(u), int(v)), 8, (255, 0, 0), 2)  # 蓝色边框
                cv2.circle(img, (int(u), int(v)), 3, (255, 255, 255), -1)  # 白色中心
                
                # 绘制相机朝向箭头
                cam_rotation = cam['rotation']
                yaw = cam_rotation[1]  # 使用yaw角度
                
                # 计算箭头终点（修正角度转换）
                arrow_length = 25
                # Unreal坐标系：yaw=0指向X轴正方向，在图像中对应向上
                # 所以图像角度 = yaw - 90°
                image_yaw_rad = math.radians(yaw - 90)
                arrow_end_x = u + arrow_length * math.cos(image_yaw_rad)
                arrow_end_y = v + arrow_length * math.sin(image_yaw_rad)
                
                cv2.arrowedLine(img, (int(u), int(v)), (int(arrow_end_x), int(arrow_end_y)), 
                            (255, 0, 0), 2, tipLength=0.3)
                
                # 添加相机ID标签
                label_text = f"C{cam['id']}"
                text_x, text_y = self.get_smart_text_position(u + 40, v + 40, W, H, label_text)
                cv2.putText(img, label_text, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    def draw_next_object_preview(self, img, start_x, preview_width, img_height, next_object, real_scale):
        """
        在右侧预览区域绘制下一个待放置物体的1:1预览
        """
        import cv2
        import numpy as np
        import math
        
        # 预览区域参数
        preview_start_x = start_x + 20
        preview_center_x = start_x + preview_width // 2
        preview_center_y = img_height // 2
        
        # 获取物体信息
        name = next_object.get('name', 'Unknown')
        length = next_object.get('length', 100)
        width = next_object.get('width', 100)
        agent_type = next_object.get('agent_type', 'unknown')
        rotation = next_object.get('rotation', [0, 0, 0])
        yaw = rotation[1] if len(rotation) > 1 else 0
        
        # 计算真实像素尺寸
        obj_length_px = length * real_scale
        obj_width_px = width * real_scale
        
        # 标题区域
        cv2.putText(img, "Next Object Preview", (preview_start_x, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # 物体信息文本
        info_y = 60
        cv2.putText(img, f"Name: {name}", (preview_start_x, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        info_y += 25
        cv2.putText(img, f"Size: {length}x{width} cm", (preview_start_x, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        info_y += 25
        cv2.putText(img, f"Type: {agent_type}", (preview_start_x, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        info_y += 25
        cv2.putText(img, f"Rotation: {yaw:.1f} degrees", (preview_start_x, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        info_y += 25
        cv2.putText(img, f"Scale: 1cm = {real_scale:.2f}px", (preview_start_x, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # 物体类型颜色
        type_colors = {
            'car': (50, 50, 200),        # 深红
            'motorbike': (50, 200, 50),  # 深绿
            'player': (200, 50, 50),     # 深蓝
            'animal': (50, 200, 200),    # 深黄
            'drone': (200, 50, 200)      # 深紫
        }
        color = type_colors.get(agent_type, (100, 100, 100))
        
        # 确保物体能在预览区域内显示
        max_dimension = max(obj_length_px, obj_width_px)
        available_space = min(preview_width - 40, img_height - info_y - 100)
        
        if max_dimension > available_space:
            # 如果物体太大，按比例缩小但保持1:1比例关系
            scale_factor = available_space / max_dimension
            display_length_px = obj_length_px * scale_factor
            display_width_px = obj_width_px * scale_factor
            
            # 显示缩放信息
            info_y += 20
            cv2.putText(img, f"Preview scaled {scale_factor:.2f}x", (preview_start_x, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 100, 0), 1)
            cv2.putText(img, "(to fit in preview area)", (preview_start_x, info_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 100, 0), 1)
        else:
            # 使用真实尺寸
            display_length_px = obj_length_px
            display_width_px = obj_width_px
            
            # 显示真实尺寸标记
            info_y += 20
            cv2.putText(img, " REAL SIZE ", (preview_start_x, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2)
        
        # 调整预览中心位置
        preview_obj_center_y = int(info_y + 80 + max(display_length_px, display_width_px) // 2)
        
        # 绘制1:1尺寸的物体
        self.draw_rotated_object_icon_real_size(img, preview_center_x, preview_obj_center_y, 
                                               display_length_px, display_width_px, yaw, color)
        
        # 绘制尺寸标注
        self.draw_size_annotations(img, preview_center_x, preview_obj_center_y, 
                                 display_length_px, display_width_px, yaw, length, width)
        
        # 绘制坐标轴指示
        self.draw_coordinate_reference(img, preview_start_x, preview_obj_center_y + max(display_length_px, display_width_px) // 2 + 50)

    def draw_coordinate_reference(self, img, start_x, start_y):
        """
        绘制坐标轴参考（修正为正确的UnrealCV坐标系）
        """
        import cv2
        
        # 确保坐标是整数类型
        start_x = int(start_x + 50)
        start_y = int(start_y + 50)
        
        # 绘制坐标轴
        axis_length = 50
        
        # X轴向前（向上）- 红色
        cv2.arrowedLine(img, (start_x, start_y), (start_x, start_y - axis_length), 
                       (0, 0, 255), 2, tipLength=0.3)
        cv2.putText(img, "X (Forward)", (start_x, start_y - axis_length - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
        # Y轴向右 - 绿色
        cv2.arrowedLine(img, (start_x, start_y), (start_x + axis_length, start_y), 
                       (0, 255, 0), 2, tipLength=0.3)
        cv2.putText(img, "Y (Right)", (start_x + axis_length + 5, start_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 坐标系说明
        cv2.putText(img, "Unreal Coords", (start_x, start_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # 添加原点标记
        cv2.circle(img, (start_x, start_y), 3, (0, 0, 0), -1)
        
        # 添加yaw角度参考
        cv2.putText(img, "Yaw=0: +X direction", (start_x, start_y + 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(img, "Clockwise: +Yaw", (start_x, start_y + 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    def draw_size_annotations(self, img, center_x, center_y, length_px, width_px, yaw_deg, real_length, real_width):
        """
        绘制物体的尺寸标注线（修正坐标系）
        """
        import cv2
        import numpy as np
        import math
        
        # 使用修正后的角度转换
        image_yaw_deg = yaw_deg - 90
        yaw_rad = math.radians(image_yaw_deg)
        
        # 计算物体的四个角
        half_length = length_px / 2.0
        half_width = width_px / 2.0
        
        # 长度方向的标注（沿着物体的长轴）
        length_start = np.array([center_x - half_length * math.cos(yaw_rad), 
                                center_y - half_length * math.sin(yaw_rad)])
        length_end = np.array([center_x + half_length * math.cos(yaw_rad), 
                            center_y + half_length * math.sin(yaw_rad)])
        
        # 绘制长度标注线（在物体外侧）
        offset_distance = max(half_width + 20, 30)
        offset_x = -offset_distance * math.sin(yaw_rad)
        offset_y = offset_distance * math.cos(yaw_rad)
        
        length_line_start = length_start + np.array([offset_x, offset_y])
        length_line_end = length_end + np.array([offset_x, offset_y])
        
        # 绘制标注线
        cv2.line(img, tuple(length_line_start.astype(int)), tuple(length_line_end.astype(int)), (0, 0, 0), 1)
        
        # 绘制端点短线
        perp_x = 10 * math.sin(yaw_rad)
        perp_y = -10 * math.cos(yaw_rad)
        
        cv2.line(img, tuple((length_line_start + np.array([perp_x, perp_y])).astype(int)), 
                tuple((length_line_start - np.array([perp_x, perp_y])).astype(int)), (0, 0, 0), 1)
        cv2.line(img, tuple((length_line_end + np.array([perp_x, perp_y])).astype(int)), 
                tuple((length_line_end - np.array([perp_x, perp_y])).astype(int)), (0, 0, 0), 1)
        
        # 标注文字
        length_text_pos = (length_line_start + length_line_end) / 2
        cv2.putText(img, f"{real_length}cm", tuple(length_text_pos.astype(int)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # 宽度方向的标注
        width_offset_distance = max(half_length + 30, 40)
        width_offset_x = width_offset_distance * math.cos(yaw_rad)
        width_offset_y = width_offset_distance * math.sin(yaw_rad)
        
        width_start = np.array([center_x - half_width * (-math.sin(yaw_rad)), 
                            center_y - half_width * math.cos(yaw_rad)])
        width_end = np.array([center_x + half_width * (-math.sin(yaw_rad)), 
                            center_y + half_width * math.cos(yaw_rad)])
        
        width_line_start = width_start + np.array([width_offset_x, width_offset_y])
        width_line_end = width_end + np.array([width_offset_x, width_offset_y])
        
        cv2.line(img, tuple(width_line_start.astype(int)), tuple(width_line_end.astype(int)), (0, 0, 0), 1)
        
        # 宽度标注的端点短线
        width_perp_x = 10 * math.cos(yaw_rad)
        width_perp_y = 10 * math.sin(yaw_rad)
        
        cv2.line(img, tuple((width_line_start + np.array([width_perp_x, width_perp_y])).astype(int)), 
                tuple((width_line_start - np.array([width_perp_x, width_perp_y])).astype(int)), (0, 0, 0), 1)
        cv2.line(img, tuple((width_line_end + np.array([width_perp_x, width_perp_y])).astype(int)), 
                tuple((width_line_end - np.array([width_perp_x, width_perp_y])).astype(int)), (0, 0, 0), 1)
        
        # 宽度标注文字
        width_text_pos = (width_line_start + width_line_end) / 2
        cv2.putText(img, f"{real_width}cm", tuple(width_text_pos.astype(int)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    def draw_rotated_object_icon_real_size(self, img, center_x, center_y, icon_length, icon_width, yaw_deg, color):
        """
        绘制按真实像素尺寸的旋转物体图标（修正Unreal Engine坐标系）
        """
        import cv2
        import numpy as np
        import math
        
        # 确保坐标是整数类型
        center_x = int(center_x)
        center_y = int(center_y)
        
        # 确保最小尺寸
        icon_length = max(3, float(icon_length))
        icon_width = max(2, float(icon_width))
        
        # 计算四个角的相对位置
        half_length = icon_length / 2.0
        half_width = icon_width / 2.0
        
        corners = np.array([
            [-half_length, -half_width],  # 左后
            [half_length, -half_width],   # 右后
            [half_length, half_width],    # 右前
            [-half_length, half_width]    # 左前
        ])
        
        # Unreal Engine坐标系到图像坐标系的转换：
        # Unreal: X向前(北), Y向右(东), Z向上
        # 图像: X向右(东), Y向下(南)
        # 所以需要将Unreal的坐标映射到图像坐标：
        # Unreal_X -> 图像_Y的负方向 (向上)
        # Unreal_Y -> 图像_X (向右)
        
        # 在Unreal中，yaw=0时朝向X轴正方向（前方）
        # 在图像中，这应该对应向上（-Y方向）
        # 所以图像中的角度 = Unreal_yaw - 90°
        image_yaw_deg = yaw_deg - 90
        yaw_rad = math.radians(image_yaw_deg)
        
        # 标准的2D旋转矩阵（顺时针为正）
        rotation_matrix = np.array([
            [math.cos(yaw_rad), -math.sin(yaw_rad)],
            [math.sin(yaw_rad), math.cos(yaw_rad)]
        ])
        
        rotated_corners = corners @ rotation_matrix.T
        
        # 转换到图像坐标
        icon_corners = rotated_corners + np.array([center_x, center_y])
        icon_corners = icon_corners.astype(np.int32)
        
        # 绘制填充的多边形
        cv2.fillPoly(img, [icon_corners], color)
        
        # 绘制边框
        cv2.polylines(img, [icon_corners], isClosed=True, color=(0, 0, 0), thickness=1)
        
        # 绘制方向指示器（指向物体前方）
        if icon_length > 10:
            # 前方向量：在Unreal中是X轴正方向，在图像中对应-Y方向
            front_direction = np.array([math.cos(yaw_rad), math.sin(yaw_rad)])
            front_center = np.array([center_x, center_y], dtype=np.float64) + \
                        (half_length + 5) * front_direction
            arrow_tip = front_center.astype(np.int32)
            
            # 确保坐标是整数元组
            start_point = (int(center_x), int(center_y))
            end_point = (int(arrow_tip[0]), int(arrow_tip[1]))
            
            cv2.arrowedLine(img, start_point, end_point, 
                        (0, 0, 0), thickness=2, tipLength=0.4)