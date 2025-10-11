import pickle
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
from collections import deque
 

class GraphBasedSampler:
    def __init__(self, graph_pickle_file):
        """从.gpickle文件初始化采样器"""
        # 加载保存的图
        with open(graph_pickle_file, 'rb') as f:
            self.graph = pickle.load(f)
        
        # 获取节点位置
        self.node_positions = {}
        for node in self.graph.nodes():
            self.node_positions[node] = self.graph.nodes[node]['pos']
        self.historical_positions = []
        self.exclusion_radius = 200
        print(f"已加载图: {len(self.graph.nodes())}个节点, {len(self.graph.edges())}条边")

    def compute_adaptive_all_max_distance(self, agent_configs, density_factor=6, min_limit=600, max_limit=3500):
        """
        根据所有智能体的实际占地面积自适应计算all_max_distance，
        对于比车大的agent，面积权重增大
        """
        car_area = 400 * 200
        big_agent_weight = 2  # 比车大的面积权重
        total_area = 0
        for agent_type, config in agent_configs.items():
            for i in range(len(config['name'])):
                if agent_type == 'car':
                    size = (400, 200)
                elif agent_type == 'player':
                    size = (50, 50)
                elif agent_type == 'motorbike':
                    size = (180, 60)
                elif agent_type == 'drone':
                    size = (100, 100)
                elif agent_type == 'animal':
                    app_id = config['app_id'][i]
                    if app_id == 0:
                        size = (50, 50)
                    elif app_id in [1, 2]:
                        size = (80, 50)
                    elif app_id in [3, 5, 12]:
                        size = (40, 40)
                    elif app_id == 6:
                        size = (300, 150)
                    elif app_id == 9:
                        size = (30, 30)
                    elif app_id in [10, 14]:
                        size = (400, 150)
                    elif app_id in [11, 15, 25, 26]:
                        size = (300, 280)
                    elif app_id in [16, 20, 21, 22]:
                        size = (350, 280)
                    elif app_id == 19:
                        size = (250, 280)
                    elif app_id == 23:
                        size = (400, 280)
                    elif app_id == 27:
                        size = (300, 200)
                    else:
                        size = (50, 50)
                else:
                    size = (100, 100)
                area = size[0] * size[1]
                # 对比车大的agent增大权重
                if area > car_area:
                    area *= big_agent_weight
                total_area += area
        sample_area = total_area * density_factor
        all_max_distance = int(np.sqrt(sample_area / np.pi) * 2)
        all_max_distance = max(all_max_distance, min_limit)
        all_max_distance = min(all_max_distance, max_limit)
        print(f"根据总面积自适应all_max_distance={all_max_distance}")
        return all_max_distance


    def _sample_external_cameras(self, objects, camera_count=8, 
                                 # 新增或调整参数以适应新方法
                                 ring_inner_radius_offset=200, # 圆环内半径相对于智能体包围圆的偏移
                                 ring_outer_radius_offset=600, # 圆环外半径相对于智能体包围圆的偏移
                                 min_angle_separation_deg=30, 
                                 **kwargs): # 保留kwargs以获取其他可能参数
        """
        新思路：在智能体集群外的一个圆环区域内采样相机，所有相机朝向集群中心。
        """
        import math
        import numpy as np
        import random

        if not objects:
            print("没有物体可供采样相机。")
            return []

        positions_2d = np.array([obj["position"][:2] for obj in objects])
        
        if len(positions_2d) == 0:
            print("物体位置列表为空。")
            return []

        # 1. 计算智能体集群的中心和包围圆半径
        # 使用所有已放置智能体的平均位置作为集群中心
        cluster_center_2d = np.mean(positions_2d, axis=0)
        
        if len(positions_2d) == 1:
            # 如果只有一个物体，最远距离是0，需要一个基础半径
            agent_bounding_radius = kwargs.get('min_agent_radius_for_single_object', 100) 
        else:
            distances_from_center = np.linalg.norm(positions_2d - cluster_center_2d, axis=1)
            agent_bounding_radius = np.max(distances_from_center) if len(distances_from_center) > 0 else 0
        
        # 2. 定义相机采样圆环的内外半径
        # 内半径 = 智能体包围半径 + 内偏移 (确保在外部)
        # 外半径 = 智能体包围半径 + 外偏移
        # 也可考虑结合kwargs传入的 min_distance/max_distance (相对于中心)
        # 这里简化为基于 agent_bounding_radius 和 offset
        
        # 从kwargs获取相机与智能体的最小安全距离，确保圆环内径足够大
        min_cam_to_agent_dist = kwargs.get('min_cam_to_agent_dist', 150)

        # 圆环内径至少是 agent_bounding_radius + min_cam_to_agent_dist，并且加上一个额外的偏移
        # 这样保证了相机节点本身离最外层agent有min_cam_to_agent_dist的距离
        # 同时，ring_inner_radius_offset 可以理解为相机到“集群边缘”的额外距离
        camera_ring_inner_radius = agent_bounding_radius + max(min_cam_to_agent_dist, ring_inner_radius_offset)
        camera_ring_outer_radius = agent_bounding_radius + ring_outer_radius_offset

        # 确保外半径大于内半径
        if camera_ring_outer_radius <= camera_ring_inner_radius:
            camera_ring_outer_radius = camera_ring_inner_radius + 300 # 保证一个最小的环宽度
            print(f"警告：计算出的相机圆环外半径({camera_ring_outer_radius})不大于内半径({camera_ring_inner_radius})。已调整外半径。")

        print(f"智能体集群中心: {cluster_center_2d}, 包围半径: {agent_bounding_radius:.1f}")
        print(f"相机采样圆环: 内径={camera_ring_inner_radius:.1f}, 外径={camera_ring_outer_radius:.1f}")

        # 3. 在圆环内生成候选相机点
        candidate_camera_infos = []
        # 增加初始候选点的数量
        num_initial_angular_samples = max(48, camera_count * 6)
        num_radial_samples = 3 # 在圆环内部分几层采样半径

        for i in range(num_initial_angular_samples):
            angle = (2 * np.pi / num_initial_angular_samples) * i
            angle += np.random.uniform(-np.pi / (num_initial_angular_samples * 2), np.pi / (num_initial_angular_samples * 2))

            for j in range(num_radial_samples):
                # 在内外半径之间均匀或随机选择一个采样半径
                current_sample_radius = camera_ring_inner_radius + (camera_ring_outer_radius - camera_ring_inner_radius) * (j / max(1, num_radial_samples -1)) if num_radial_samples > 1 else camera_ring_inner_radius
                # 或者随机: current_sample_radius = random.uniform(camera_ring_inner_radius, camera_ring_outer_radius)


                cam_x = cluster_center_2d[0] + current_sample_radius * np.cos(angle)
                cam_y = cluster_center_2d[1] + current_sample_radius * np.sin(angle)
                
                node = self._find_nearest_valid_node((cam_x, cam_y))
                if node is not None:
                    pos_3d = self.node_positions[node]
                    
                    # 所有相机朝向集群中心 cluster_center_2d
                    dx_view = cluster_center_2d[0] - pos_3d[0]
                    dy_view = cluster_center_2d[1] - pos_3d[1]
                    view_angle_rad = math.atan2(dy_view, dx_view)
                    yaw = math.degrees(view_angle_rad)
                    
                    candidate_camera_infos.append({
                        'node': node, 
                        'position': pos_3d, 
                        'rotation': [0, yaw, 0],
                        'radius_from_center': np.linalg.norm(np.array(pos_3d[:2]) - cluster_center_2d) # 实际到中心的距离
                    })

        # 4. 筛选和选择最终相机点
        # 首先基于图节点去重 (选择离圆环理想半径更近的，或随机)
        unique_node_candidates_dict = {}
        for cam_info in candidate_camera_infos:
            node = cam_info['node']
            if node not in unique_node_candidates_dict:
                unique_node_candidates_dict[node] = cam_info
            else: 
                # 策略：例如选择更接近圆环中间的，或简单替换
                # 这里简化为保留第一个遇到的
                pass 

        filtered_candidates = list(unique_node_candidates_dict.values())
        random.shuffle(filtered_candidates)

        selected_cameras_final_info = []
        min_angle_sep_rad = np.radians(min_angle_separation_deg)
        
        # 用于角度分离检查的智能体位置 (虽然相机都朝向中心，但相机本身的位置角度需要分散)
        agent_positions_2d_for_check = np.array([obj["position"][:2] for obj in objects])


        for candidate in filtered_candidates:
            if len(selected_cameras_final_info) >= camera_count:
                break

            candidate_pos_2d = np.array(candidate['position'][:2])

            # 检查与智能体的最小距离 (虽然圆环设计上应该避免，但双重检查更安全)
            # 这一步其实可以省略，因为圆环的内径已经考虑了 min_cam_to_agent_dist
            # too_close_to_agent = False
            # if len(agent_positions_2d_for_check) > 0:
            #     distances_to_agents = np.linalg.norm(agent_positions_2d_for_check - candidate_pos_2d, axis=1)
            #     if np.any(distances_to_agents < min_cam_to_agent_dist):
            #         too_close_to_agent = True
            # if too_close_to_agent:
            #     continue

            # 检查角度分离 (基于相机位置相对于集群中心的角度)
            is_angle_ok = True
            cam_pos_relative_to_center_x = candidate_pos_2d[0] - cluster_center_2d[0]
            cam_pos_relative_to_center_y = candidate_pos_2d[1] - cluster_center_2d[1]
            current_cam_angle_rad = math.atan2(cam_pos_relative_to_center_y, cam_pos_relative_to_center_x)

            for existing_cam_info in selected_cameras_final_info:
                existing_pos_2d = np.array(existing_cam_info['position'][:2])
                existing_pos_relative_to_center_x = existing_pos_2d[0] - cluster_center_2d[0]
                existing_pos_relative_to_center_y = existing_pos_2d[1] - cluster_center_2d[1]
                existing_cam_angle_rad = math.atan2(existing_pos_relative_to_center_y, existing_pos_relative_to_center_x)
                
                angle_diff = abs(current_cam_angle_rad - existing_cam_angle_rad)
                angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                
                if angle_diff < min_angle_sep_rad:
                    is_angle_ok = False
                    break
            
            if is_angle_ok:
                selected_cameras_final_info.append(candidate)

        # 如果严格筛选后相机数量不足
        if len(selected_cameras_final_info) < camera_count:
            needed_more = camera_count - len(selected_cameras_final_info)
            potential_fillers = [c for c in filtered_candidates if not any(c['node'] == sel_cam['node'] for sel_cam in selected_cameras_final_info)]
            # 对于补充点，可以放宽角度限制，但最好还是检查一下与agent的距离（如果之前省略了）
            fillers_to_add = []
            for cand_fill in potential_fillers:
                if len(fillers_to_add) >= needed_more:
                    break
                # (如果需要，在这里补充 min_cam_to_agent_dist 检查)
                fillers_to_add.append(cand_fill)
            selected_cameras_final_info.extend(fillers_to_add)


        # 5. 格式化输出
        output_cameras = []
        for i, cam_info in enumerate(selected_cameras_final_info):
            output_cameras.append({
                'id': i,
                'position': cam_info['position'],
                'rotation': cam_info['rotation'],
                'node': cam_info['node'],
                'visible_objects': [], 
                'coverage': 0          
            })

        print(f"新思路相机采样完成，共获得 {len(output_cameras)} 个相机点。")
        return output_cameras
    
    def _determine_object_orientation(self, node, obj_type):
        """基于环境约束确定物体的朝向"""
        import math
        import random
        import numpy as np
        
        # 获取节点的邻居，用于推断可能的路径方向
        neighbors = list(self.graph.neighbors(node))
        
        # 如果有邻居节点，使用它们来推断方向
        if neighbors:
            # 计算当前节点到所有邻居的方向向量
            directions = []
            node_pos = np.array(self.node_positions[node])
            
            for neighbor in neighbors:
                neighbor_pos = np.array(self.node_positions[neighbor])
                direction = neighbor_pos - node_pos
                
                # 只考虑有意义的距离
                if np.linalg.norm(direction) > 1e-6:
                    # 计算方向角度 (atan2给出的是数学坐标系中的角度)
                    angle_rad = math.atan2(direction[1], direction[0])
                    # 将数学角度转换为UnrealZoo的yaw角度 (左手系)
                    yaw = -math.degrees(angle_rad)
                    
                    # 确保yaw在-180到180范围内
                    if yaw > 180:
                        yaw -= 360
                    elif yaw < -180:
                        yaw += 360
                    
                    directions.append(yaw)
            
            # 根据物体类型选择合适的朝向策略
            if obj_type in ['car',  'motorbike', 'drone']:
                # 车辆更倾向于沿路径方向
                if directions:
                    # 选择一个方向，并添加小的随机偏差
                    base_yaw = random.choice(directions)
                    return base_yaw + random.uniform(-15, 15)  # 添加小的随机偏差
            
            elif obj_type in ['human', 'animal']:
                # 人和动物朝向可以更随机
                if directions:
                    # 可以选择任何邻居方向，或者这些方向的对立面
                    base_yaw = random.choice(directions)
                    # 50%的概率反向
                    if random.random() < 0.5:
                        base_yaw = (base_yaw + 180) % 360
                        if base_yaw > 180:
                            base_yaw -= 360
                    return base_yaw + random.uniform(-45, 45)  # 添加更大的随机偏差
        
        # 如果没有可用的邻居或未定义类型，使用随机朝向
        return random.uniform(-180, 180)

    def _can_place_object_at(self, node, occupied_areas, length, width, min_distance):
        """检查节点是否可以放置物体，考虑旋转角度"""
        import math
        import numpy as np
        
        if node not in self.node_positions:
            return False
            
        node_pos = np.array(self.node_positions[node])
        
        # 检查是否与已占用区域重叠或过近
        for occ_node, occ_length, occ_width, occ_rotation in occupied_areas:
            occ_pos = np.array(self.node_positions[occ_node])
            
            # 简化检查 - 使用旋转后的边界框进行近似检查
            # 计算两个中心点之间的距离
            center_dist = np.linalg.norm(node_pos[:2] - occ_pos[:2])
            
            # 计算两个物体的"半径"(考虑旋转后的最大尺寸)
            radius1 = math.sqrt((length/2)**2 + (width/2)**2)
            radius2 = math.sqrt((occ_length/2)**2 + (occ_width/2)**2)
            
            # 检查碰撞
            if center_dist < radius1 + radius2 + min_distance:
                return False
        
        return True
    

    def _mark_area_occupied(self, node, occupied_areas, valid_mask,length, width, rotation=0):
        """标记被物体占用的区域，考虑旋转角度"""
        occupied_areas.append((node, length, width, rotation))
        return valid_mask

 
    def _find_nearest_valid_node(self, target_pos, obj_node=None, max_search=20):
        """找到最接近目标位置的有效节点"""
        nodes = list(self.node_positions.keys())
        positions = np.array(list(self.node_positions.values()))
        
        # 计算到目标的距离
        dists = np.linalg.norm(positions[:, :2] - np.array(target_pos), axis=1)
        
        # 获取最近的几个节点
        nearest_indices = np.argsort(dists)[:max_search]
        
        # 找到第一个有效节点
        for idx in nearest_indices:
            node = nodes[idx]
            if (obj_node is None or node != obj_node) and node in self.graph:
                return node
                
        return None
    

    def sample_for_predefined_agents(self, agent_configs, min_edge_distance=50, 
                                    max_center_distance=700, camera_count=3, 
                                    max_steps=5000, vehicle_zones=None, all_max_distance=None, **kwargs): # 添加 **kwargs
        """
        更高内聚度采样：先定中心点，再在圆形区域内采样所有物体。
        现在会将相机采样相关的kwargs传递给_sample_external_cameras。
        """
        def _all_within_max_distance(new_node, sampled_objects, max_distance):
            if not sampled_objects or max_distance is None:
                return True
            new_pos = np.array(self.node_positions[new_node])
            for obj in sampled_objects:
                obj_pos = np.array(self.node_positions[obj["node"]])
                dist = np.linalg.norm(new_pos - obj_pos)
                if dist > max_distance:
                    return False
            return True
    
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

        if not object_list: 
            print("没有待采样的物体。")
            return {}, {}

        # 2. 预处理车辆区域节点
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
    
        # 3. 采样中心点
        all_nodes = list(self.graph.nodes())
        if not all_nodes:
            print("错误：图中没有节点可供采样。")
            # 返回空配置和None的中心/半径
            return {'agent_configs': {}, 'camera_configs': {}, 'sampling_center': None, 'sampling_radius': None}


        center_node_source = "图的随机节点"
        if all_objects_are_small: # all_objects_are_small 和 has_car 在步骤1中计算
            center_node = random.choice(all_nodes)
            center_node_source = "图的随机节点 (所有物体较小)"
        elif has_car and vehicle_zone_nodes.get('car'):
            center_node = random.choice(vehicle_zone_nodes['car'])
            center_node_source = "车辆区域的随机节点 (因有车)"
        else:
            center_node = random.choice(all_nodes)
        
        print(f"中心点从 {center_node_source} 选择。")
        # agent_sampling_center_node = center_node # 可以保留节点ID如果需要
        agent_sampling_center_pos = np.array(self.node_positions[center_node]) # <--- 这是智能体采样的圆心位置
    
        # 4. 采样半径
        if all_max_distance is not None:
            agent_sampling_radius = all_max_distance / 2 # <--- 这是智能体采样的半径
        else:
            agent_sampling_radius = 1500  # 默认
    
        # 5. 采样区域内的所有节点
        candidate_nodes = [
            node for node in all_nodes
            if np.linalg.norm(np.array(self.node_positions[node]) - agent_sampling_center_pos) <= agent_sampling_radius
        ]
        if not candidate_nodes:
            print(f"在中心点 {center_node} 半径 {agent_sampling_radius} 内没有足够的候选节点。请增大all_max_distance或检查图。")
            # 返回空配置和计算出的中心/半径
            return {'agent_configs': {}, 
                    'camera_configs': {}, 
                    'sampling_center': agent_sampling_center_pos.tolist() if isinstance(agent_sampling_center_pos, np.ndarray) else agent_sampling_center_pos, 
                    'sampling_radius': agent_sampling_radius}
    
        # 6. 采样主循环
        sampled_objects = []
        occupied_areas = [] # (node, length, width, rotation)
        steps = 0
    
        for obj_info in object_list:
            agent_type, length, width, name, app_id, animation,feature_caption,type = obj_info
            placed = False
            local_steps = 0
            max_local_steps = 500 
    
            use_vehicle_zone_for_this_agent = False
            if agent_type in vehicle_zone_nodes: # 如果该类型的区域存在
                use_vehicle_zone_for_this_agent = True
            
            current_valid_nodes_for_agent = []
            if use_vehicle_zone_for_this_agent:
                # 取采样区域内节点与该类型车辆区域节点的交集
                current_valid_nodes_for_agent = list(set(candidate_nodes) & set(vehicle_zone_nodes[agent_type]))
            else:
                current_valid_nodes_for_agent = candidate_nodes
    
            if not current_valid_nodes_for_agent:
                print(f"没有可用节点可供 {agent_type} ({name}) 采样（区域限制后）。跳过此物体。")
                continue 
            
            # 尝试从一个随机的有效节点开始
            current_node_to_try = random.choice(current_valid_nodes_for_agent)
    
            while not placed and local_steps < max_local_steps and steps < max_steps:
                if self._can_place_object_at(current_node_to_try, occupied_areas, length, width, min_edge_distance):
                    # _check_center_distance 和 _all_within_max_distance 的逻辑可以根据需要保留或调整
                    # if self._check_center_distance(current_node_to_try, sampled_objects, max_center_distance):
                    #    if _all_within_max_distance(current_node_to_try, sampled_objects, all_max_distance):
                    yaw = self._determine_object_orientation(current_node_to_try, agent_type)
                    position = self.node_positions[current_node_to_try]
                    rotation = [0, yaw, 0]
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
                    self._mark_area_occupied(current_node_to_try, occupied_areas, length, width, yaw)
                    placed = True
                    break 
    
                # 移动到下一个节点 (简化：从有效节点中随机选一个不同的)
                # 更复杂的可以是BFS/DFS或基于距离的搜索
                next_nodes_options = [n for n in current_valid_nodes_for_agent if n != current_node_to_try]
                if next_nodes_options:
                    current_node_to_try = random.choice(next_nodes_options)
                else: # 如果只有一个有效节点且不满足条件，则无法放置
                    break

                local_steps += 1
                steps += 1
    
            if not placed:
                print(f"警告: 无法为 {name} ({agent_type}) 找到合适位置。")
    
        # 7. 采样相机 (将kwargs传递下去)
        cameras = self._sample_external_cameras(
            objects=sampled_objects, 
            camera_count=camera_count,
            # 传递新方法所需的参数，或依赖其默认值
            ring_inner_radius_offset=kwargs.get('ring_inner_radius_offset', 200), 
            ring_outer_radius_offset=kwargs.get('ring_outer_radius_offset', 800),
            min_angle_separation_deg=kwargs.get('min_angle_separation_deg', 30),
            min_cam_to_agent_dist=kwargs.get('min_cam_to_agent_dist', 150) # 仍然重要，用于定义圆环内边界
            # fov_deg 和 desired_closeness_factor 在这个新方法中作用不大
        )
    
        # 8. 转换为配置格式
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
    
        print(f"成功为 {len(sampled_objects)}/{len(object_list)} 个代理找到位置，总步数: {steps}")
        if len(sampled_objects) < len(object_list):
            print(f"注意：并非所有请求的代理都成功放置。请求数: {len(object_list)}, 成功数: {len(sampled_objects)}")
        
        # 将 NumPy 数组转换为列表以便序列化（如果需要）
        center_pos_to_return = agent_sampling_center_pos.tolist() if isinstance(agent_sampling_center_pos, np.ndarray) else agent_sampling_center_pos
        
        # 返回包含所有信息的字典
        return {
            'agent_configs': updated_configs,
            'camera_configs': camera_configs,
            'sampling_center': center_pos_to_return,
            'sampling_radius': agent_sampling_radius
        }

def visualize_with_vehicle_zones(objects, cameras, node_positions, object_types, 
                                vehicle_zones=None, output_file="cluster_with_zones.png"):
    """可视化物体、集群相机位置和车辆区域限制"""
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon, FancyArrow
    from matplotlib.lines import Line2D
    import matplotlib.path as mpath
    
    plt.figure(figsize=(16, 16))
    plt.gca().invert_yaxis()
    
    # 绘制背景点
    all_pos = np.array(list(node_positions.values()))
    plt.scatter(all_pos[:, 0], all_pos[:, 1], c='lightgray', s=1, alpha=0.2)
    
    # 首先绘制车辆区域限制
    if vehicle_zones:
        for vehicle_type, zones in vehicle_zones.items():
            for zone in zones:
                # 创建多边形区域
                poly = Polygon(zone, linewidth=2, 
                             edgecolor='darkgreen', facecolor='lightgreen', alpha=0.2)
                plt.gca().add_patch(poly)
                
                # 在区域中心添加标签
                zone_array = np.array(zone)
                center_x = np.mean(zone_array[:, 0])
                center_y = np.mean(zone_array[:, 1])
                plt.text(center_x, center_y, f"{vehicle_type} zone", 
                       fontsize=12, ha='center', va='center', color='darkgreen')
    
    # 为不同物体类型设置颜色
    colors = {
        'car': 'red',
        'human': 'blue',
        'animal': 'orange',
        'player': 'blue',
        'drone': 'cyan',
        'motorbike': 'magenta'
    }
    
    # 物体计数
    type_counts = {}
    
    # 绘制物体
    for i, obj in enumerate(objects):
        # 获取物体信息
        if isinstance(obj, dict) and "position" in obj:
            # 直接使用字典格式
            pos = obj["position"]
            obj_type = obj["type"]
            rotation = obj.get("rotation", [0, 0, 0])
            yaw = rotation[1]
        else:
            # 从更新后的配置中提取
            agent_type = list(obj.keys())[0]
            config = obj[agent_type]
            if i < len(config.get('start_pos', [])):
                pos = config['start_pos'][i][:3]  # 提取x,y,z
                yaw = config['start_pos'][i][4]   # 提取yaw
                obj_type = agent_type
            else:
                continue
        
        # 获取物体尺寸
        if obj_type in object_types:
            length, width = object_types[obj_type]
        else:
            # 根据类型设置默认尺寸
            if obj_type == 'car':
                length, width = 400, 200
            elif obj_type == 'player':
                length, width = 50, 50
            elif obj_type == 'motorbike':
                length, width = 180, 60
            elif obj_type == 'drone':
                length, width = 100, 100
            elif obj_type == 'animal':
                length, width = 80, 50
            else:
                length, width = 100, 100
            
        # 物体颜色
        if obj_type not in colors:
            colors[obj_type] = np.random.rand(3,)
        
        # 绘制物体中心点
        plt.scatter(pos[0], pos[1], c=colors[obj_type], s=80, alpha=0.8)
        
        # 标注物体编号
        plt.annotate(f"{i}", (pos[0], pos[1]), fontsize=12)
        
        # 考虑旋转绘制物体轮廓
        yaw_rad = math.radians(-yaw)  # 转换为弧度，适应UnrealZoo左手系
        
        # 创建矩形的四个角点
        corners = [
            [-length/2, -width/2],
            [length/2, -width/2],
            [length/2, width/2],
            [-length/2, width/2]
        ]
        
        # 应用旋转
        rotated_corners = []
        for x, y in corners:
            x_rot = x * math.cos(yaw_rad) + y * math.sin(yaw_rad)
            y_rot = -x * math.sin(yaw_rad) + y * math.cos(yaw_rad)
            rotated_corners.append([pos[0] + x_rot, pos[1] + y_rot])
        
        # 绘制旋转后的多边形
        polygon = Polygon(rotated_corners, linewidth=1.5, 
                         edgecolor=colors[obj_type], facecolor='none', alpha=0.7)
        plt.gca().add_patch(polygon)
        
        # 绘制朝向指示线
        front_len = length/2 * 0.8  # 前方指示线长度
        front_x = pos[0] + front_len * math.cos(yaw_rad)
        front_y = pos[1] - front_len * math.sin(yaw_rad)
        plt.plot([pos[0], front_x], [pos[1], front_y], 
                color=colors[obj_type], linewidth=2, alpha=0.8)
        
        # 更新计数
        if obj_type not in type_counts:
            type_counts[obj_type] = 0
        type_counts[obj_type] += 1

    # 绘制相机位置和朝向
        # 绘制相机位置和朝向
    if cameras and "camera" in cameras:
        cam_cfg = cameras["camera"]
        for i, cam_pos in enumerate(cam_cfg["start_pos"]):
            x, y = cam_pos[0], cam_pos[1]
            yaw = -cam_pos[4]  # 直接用rotation里的yaw
            arrow_len = 200
            dx = arrow_len * np.cos(np.radians(yaw))
            dy = -arrow_len * np.sin(np.radians(yaw))
            plt.scatter(x, y, c='black', s=120, marker='^', label='Camera' if i == 0 else "")
            plt.arrow(x, y, dx, dy, width=20, head_width=80, head_length=80, 
                      fc='black', ec='black', alpha=0.7, length_includes_head=True)
            plt.annotate(f"C{i}", (x, y), fontsize=14, color='black', weight='bold')
    # 创建图例
    legend_elements = []
    for obj_type, count in type_counts.items():
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor=colors[obj_type], markersize=10, 
                  label=f'{obj_type} ({count})')
        )
    # 添加相机图例
    legend_elements.append(
        Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=12, label='Camera')
    )
    # 添加车辆区域图例
    if vehicle_zones:
        legend_elements.append(
            Line2D([0], [0], color='darkgreen', lw=2, alpha=0.6,
                  label='Vehicle Zone')
        )
    
    plt.title(f"Objects with Vehicle Zones - {sum(type_counts.values())} objects")
    plt.legend(handles=legend_elements)
    plt.axis('equal')
    plt.grid(alpha=0.3)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存可视化到 {output_file}")

if __name__ == "__main__":
    
    env = "SuburbNeighborhood_Day"
    point_sampler = GraphBasedSampler(f"./points_graph/{env}/environment_graph.gpickle")

    
    vehicle_zones = {
    'car': [
        # 道路区域
        [[-3500, -170], [-3500, 170], [1600, 170], [1600, -170]]
    ]
    }

    # 定义物体类型和尺寸
    object_types = {
        'car': (400, 200),
        'player': (50, 50),
        'animal': (80, 50),
        'drone': (100, 100),
        'motorbike': (180, 60)
    }
    # 预定义的agent配置
    specific_counts = {
        "player": 5,
        #"motorbike":2,
        "animal": 2,
        'car': 2
    }
    from agent_sampler import AgentSampler
    agent_sampler = AgentSampler()
    agent_configs, _ = agent_sampler.sample_with_specific_counts_no_repeat(specific_counts)
    print(f"采样的代理配置: {agent_configs}")
    all_max_distance = point_sampler.compute_adaptive_all_max_distance(agent_configs)
    print(f"计算的自适应最大距离: {all_max_distance}")
    max_retry = 200
    sampling_results = None # 初始化为 None
    for attempt in range(max_retry):
        # 调用修改后的函数
        results_dict = point_sampler.sample_for_predefined_agents( # 修改接收变量
            agent_configs=agent_configs,
            camera_count=8,
            vehicle_zones=vehicle_zones,
            all_max_distance=all_max_distance,
            ring_inner_radius_offset=300, 
            ring_outer_radius_offset=500,
            min_angle_separation_deg=35,
            min_cam_to_agent_dist=200 
        )
        
        # 从字典中提取配置
        updated_configs = results_dict['agent_configs']
        camera_configs = results_dict['camera_configs']
        agent_sampling_center = results_dict['sampling_center']
        agent_sampling_radius = results_dict['sampling_radius']
        sampling_results = results_dict # 保存完整结果

        # 检查是否所有物体都采样成功
        total_needed = sum(len(cfg['name']) for cfg in agent_configs.values())
        total_found = sum(len(cfg['name']) for cfg in updated_configs.values())
        if total_found == total_needed:
            print(f"采样成功（第{attempt+1}次）")
            print(f"智能体采样中心: {agent_sampling_center}, 采样半径: {agent_sampling_radius}")
            break
        else:
            print(f"采样失败，重新尝试（第{attempt+1}次）")
    else:
        print("多次尝试后仍未采样成功，请检查参数或环境！")
        if sampling_results: # 即使失败，也打印最后一次尝试的信息
            print(f"最后一次尝试的智能体采样中心: {sampling_results['sampling_center']}, 采样半径: {sampling_results['sampling_radius']}")


    if sampling_results and sampling_results['agent_configs']: # 确保采样结果有效
        updated_configs = sampling_results['agent_configs']
        camera_configs = sampling_results['camera_configs']
        
        objects_list = []
        for agent_type, config in updated_configs.items():
            if "start_pos" in config:
                for i, pos in enumerate(config["start_pos"]):
                    objects_list.append({
                        "type": agent_type,
                        "position": pos[:3],
                        "rotation": [0, pos[4], 0],
                        "name": config["name"][i] if i < len(config.get("name", [])) else f"{agent_type}_{i}"
                    })

        # 可视化结果
        visualize_with_vehicle_zones(
            objects_list, camera_configs, point_sampler.node_positions, object_types,
            vehicle_zones=vehicle_zones, output_file="vehicles_in_zones.png"
        )
    else:
        print("没有有效的采样结果可供可视化。")
