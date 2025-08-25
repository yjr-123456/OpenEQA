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
        
        print(f"已加载图: {len(self.graph.nodes())}个节点, {len(self.graph.edges())}条边")
    def sample_objects_with_cluster_cameras(self, object_types, num_objects=20, min_edge_distance=100, 
                                     max_center_distance=1500, camera_count=3, max_steps=5000):
        """
        先采样物体位置，然后为整个物体集群采样少量相机拍摄点
        
        Args:
            object_types: 字典 {类型名: (长, 宽)} 定义不同物体类型和尺寸
            num_objects: 要采样的物体总数
            min_edge_distance: 物体边缘之间的最小距离(cm)
            max_center_distance: 物体中心之间的最大距离(cm)
            camera_count: 要为整个集群采样的相机数量
            max_steps: 最大尝试步数
        
        Returns:
            (sampled_objects, camera_positions)
        """
        # 1. 首先使用现有方法仅采样物体位置
        sampled_objects = self.sample_objects_random_walk(
            object_types=object_types,
            num_objects=num_objects,
            min_edge_distance=min_edge_distance,
            max_center_distance=max_center_distance,
            max_steps=max_steps
        )
        
        # 2. 为整个物体集群采样相机位置
        camera_positions = self._sample_external_cameras(
            sampled_objects, 
            camera_count=camera_count
        )
        
        print(f"成功放置 {len(sampled_objects)} 个物体")
        print(f"为整个集群采样了 {len(camera_positions)} 个相机拍摄点")
        
        return sampled_objects, camera_positions

    def _sample_external_cameras(self, objects, camera_count=3, min_distance=800, max_distance=1500):
        """
        在物体聚类外部采样相机位置，使用最远点采样确保视角多样性
        
        Args:
            objects: 物体列表 [(node_id, position, type), ...]
            camera_count: 要采样的相机数量
            min_distance: 相机到聚类中心的最小距离
            max_distance: 相机到聚类中心的最大距离
            
        Returns:
            采样的相机位置列表
        """
        import math
        import numpy as np
        from scipy.spatial import ConvexHull
        
        if not objects:
            return []
        
        # 计算物体聚类的中心和边界
        positions = np.array([pos for _, pos, _ in objects])
        cluster_center = np.mean(positions, axis=0)
        
        # 计算物体聚类的凸包（用于确定内部区域）
        try:
            hull = ConvexHull(positions[:, :2])  # 只使用x,y坐标
            hull_points = positions[hull.vertices, :2]
            
            # 扩展凸包，确保相机不会太靠近物体
            expanded_hull = []
            for i in range(len(hull_points)):
                p1 = hull_points[i]
                p2 = hull_points[(i+1) % len(hull_points)]
                
                # 计算边的法向量（外向）
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length = math.sqrt(dx*dx + dy*dy)
                if length > 0:
                    nx = -dy / length  # 外法向X分量
                    ny = dx / length   # 外法向Y分量
                    
                    # 沿法向扩展
                    buffer_distance = min_distance * 0.5
                    expanded_hull.append((p1[0] + nx * buffer_distance, p1[1] + ny * buffer_distance))
            
            expanded_hull = np.array(expanded_hull)
        except:
            # 凸包计算失败，使用简单的圆形边界代替
            expanded_hull = None
            print("无法计算凸包，使用基于距离的外部采样")
        
        # 生成候选相机位置（在外部一圈）
        camera_candidates = {}
        angle_count = max(72, camera_count * 24)  # 大幅增加角度采样密度，确保有足够多的候选点
        
        # 为每个角度生成多个距离的候选点
        for i in range(angle_count):
            angle = 360 * i / angle_count
            angle_rad = math.radians(angle)
            
            # 每个角度尝试多个距离
            for dist_factor in np.linspace(1.0, 2.0, 8):  # 增加距离采样级别到8个
                distance = min_distance * dist_factor
                if distance > max_distance:
                    continue
                
                # 计算候选位置
                camera_x = cluster_center[0] + distance * math.cos(angle_rad)
                camera_y = cluster_center[1] + distance * math.sin(angle_rad)
                
                # 找到最近的有效导航节点
                camera_node = self._find_nearest_valid_node((camera_x, camera_y))
                if not camera_node:
                    continue
                
                camera_pos = self.node_positions[camera_node]
                
                # 检查是否在扩展凸包内部（如果有凸包）
                if expanded_hull is not None:
                    from matplotlib.path import Path
                    path = Path(expanded_hull)
                    if path.contains_point((camera_pos[0], camera_pos[1])):
                        continue  # 跳过内部点
                else:
                    # 检查与任何物体的距离是否太近
                    too_close = False
                    for _, obj_pos, _ in objects:
                        dist = np.sqrt((camera_pos[0]-obj_pos[0])**2 + (camera_pos[1]-obj_pos[1])**2)
                        if dist < min_distance * 0.8:  # 稍微放宽距离要求
                            too_close = True
                            break
                    if too_close:
                        continue
                
                # 计算可见物体
                visible_objects = []
                for obj_idx, (_, obj_pos, _) in enumerate(objects):
                    if self._check_clear_line_of_sight_direct(camera_pos, obj_pos, objects, obj_idx):
                        visible_objects.append(obj_idx)
                
                # 添加候选相机，记录角度信息用于均匀分布
                camera_candidates.append({
                    'node': camera_node,
                    'position': camera_pos,
                    'angle': angle,
                    'visible_objects': visible_objects,
                    'score': len(visible_objects),
                    'angle_rad': angle_rad,  # 保存弧度角用于后续计算
                })
        
        # 相机之间的最小角度和距离约束
        min_angle_between_cameras = 2 * np.pi / (camera_count + 1)  # 确保相机间角度至少为扇区宽度
        min_distance_between_cameras = min_distance * 0.8  # 相机之间的最小距离
        
        # 强制执行多样性的相机选择
        selected_cameras = []
        covered_objects = set()
        
        # 第一步：优先选择能看到最多物体的相机作为初始点
        if camera_candidates:
            best_cam = max(camera_candidates, key=lambda x: len(x['visible_objects']))
            selected_cameras.append(best_cam)
            covered_objects.update(best_cam['visible_objects'])
            camera_candidates.remove(best_cam)
        
        # 第二步：按角度间隔选择剩余相机
        while len(selected_cameras) < camera_count and camera_candidates:
            best_cam = None
            best_score = -1
            
            for cam in camera_candidates[:]:
                # 检查与已选相机的角度是否满足最小间隔要求
                angle_constraint_met = True
                distance_constraint_met = True
                
                for selected_cam in selected_cameras:
                    # 角度约束检查
                    angle_diff = min(
                        abs(cam['angle_rad'] - selected_cam['angle_rad']),
                        2*np.pi - abs(cam['angle_rad'] - selected_cam['angle_rad'])
                    )
                    if angle_diff < min_angle_between_cameras:
                        angle_constraint_met = False
                        break
                    
                    # 距离约束检查
                    dist = np.sqrt(
                        (cam['position'][0] - selected_cam['position'][0])**2 + 
                        (cam['position'][1] - selected_cam['position'][1])**2
                    )
                    if dist < min_distance_between_cameras:
                        distance_constraint_met = False
                        break
                
                # 如果不满足硬约束，跳过此候选点
                if not angle_constraint_met or not distance_constraint_met:
                    continue
                
                # 计算能新增覆盖的物体数量
                new_objects = set(cam['visible_objects']) - covered_objects
                
                # 计算与已选相机的平均角度差异（鼓励更大的差异）
                total_angle_diff = 0
                for selected_cam in selected_cameras:
                    angle_diff = min(
                        abs(cam['angle_rad'] - selected_cam['angle_rad']),
                        2*np.pi - abs(cam['angle_rad'] - selected_cam['angle_rad'])
                    )
                    total_angle_diff += angle_diff
                
                avg_angle_diff = total_angle_diff / len(selected_cameras) if selected_cameras else np.pi
                
                # 综合评分：角度差异 + 物体覆盖
                score = (avg_angle_diff / np.pi) * 15 + len(new_objects) * 2 + len(cam['visible_objects']) * 0.5
                
                if score > best_score:
                    best_score = score
                    best_cam = cam
            
            # 如果找不到满足约束的相机，逐步放宽约束
            if best_cam is None and len(selected_cameras) < camera_count:
                print(f"警告：找不到满足约束的相机，放宽角度约束...")
                min_angle_between_cameras *= 0.7  # 降低最小角度约束
                continue
                
            if best_cam:
                selected_cameras.append(best_cam)
                covered_objects.update(best_cam['visible_objects'])
                camera_candidates.remove(best_cam)
            else:
                break
                
        # 如果相机数量仍然不足，选择可见物体最多的候选点，忽略角度约束
        if len(selected_cameras) < camera_count and camera_candidates:
            print(f"警告：放宽所有约束，选择剩余相机...")
            
            # 按可见物体数量排序
            remaining_candidates = sorted(
                camera_candidates, 
                key=lambda x: len(x['visible_objects']), 
                reverse=True
            )
            
            # 添加剩余的相机
            for cam in remaining_candidates:
                if len(selected_cameras) >= camera_count:
                    break
                selected_cameras.append(cam)
                covered_objects.update(cam['visible_objects'])
        
        # 格式化结果 
        camera_positions = []
        for i, cam in enumerate(selected_cameras):
            # 计算相机应该朝向的目标点
            target_pos = cluster_center  # 默认朝向聚类中心
            
            # 计算朝向角度（使相机朝向中心）
            dx = target_pos[0] - cam['position'][0] 
            dy = target_pos[1] - cam['position'][1]
            # 使用负值将标准数学坐标系转换为UnrealZoo左手坐标系
            yaw = -math.degrees(math.atan2(dy, dx))
            
            # 确保yaw在-180到180范围内
            if yaw > 180:
                yaw -= 360
            elif yaw < -180:
                yaw += 360
            
            camera_positions.append({
                'position': cam['position'],
                'rotation': [0, yaw, 0], # pitch, yaw, roll
                'node': cam['node'],
                'visible_objects': cam['visible_objects'],
                'coverage': len(cam['visible_objects'])
            })

        uncovered_count = len(set(range(len(objects))) - covered_objects)
        print(f"采样了{len(camera_positions)}个外部相机，覆盖了{len(covered_objects)}/{len(objects)}个物体，{uncovered_count}个物体未被覆盖")
        return camera_positions

   
    
    def sample_objects_random_walk(self, object_types, num_objects=20, min_edge_distance=100, 
                                  max_center_distance=1500, max_steps=5000):
        """
        使用随机游走策略采样不同尺寸的物体放置点
        
        Args:
            object_types: 字典 {类型名: (长, 宽)} 定义不同物体类型和尺寸
            num_objects: 要采样的物体总数
            min_edge_distance: 物体边缘之间的最小距离(cm)
            max_center_distance: 物体中心之间的最大距离(cm)
            max_steps: 最大尝试步数
            
        Returns:
            采样结果 [(node_id, position, object_type), ...]
        """
        # 创建物体分配列表
        object_list = []
        for obj_type, (length, width) in object_types.items():
            # 根据面积分配数量 - 面积越大的物体数量越少
            area = length * width
            count = max(1, int(num_objects * (1000 / (area + 1000))))
            for _ in range(count):
                object_list.append((obj_type, length, width))
        
        # 调整物体总数
        random.shuffle(object_list)
        if len(object_list) > num_objects:
            object_list = object_list[:num_objects]
        else:
            # 如果物体不足，添加更多小物体
            while len(object_list) < num_objects:
                smallest_type = min(object_types.items(), key=lambda x: x[1][0] * x[1][1])
                object_list.append((smallest_type[0], smallest_type[1][0], smallest_type[1][1]))
        
        # 按面积从大到小排序物体，优先放置大物体
        object_list.sort(key=lambda x: x[1] * x[2], reverse=True)
        
        # 已采样的物体
        sampled_objects = []
        # 已占用的区域
        occupied_areas = []
        
        # 选择起始节点 (优先选择高度中心性节点)
        degrees = dict(self.graph.degree())
        candidates = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:50]
        start_node = random.choice(candidates)[0]
        
        # 初始化随机游走
        current_node = start_node
        steps = 0
        
        # 开始随机游走采样
        for obj_type, length, width in object_list:
            placed = False
            local_steps = 0
            max_local_steps = 500  # 每个物体的最大尝试次数
            
            while not placed and local_steps < max_local_steps and steps < max_steps:
                # 检查当前节点是否适合放置物体
                if self._can_place_object_at(current_node, occupied_areas, length, width, min_edge_distance):
                    # 检查与其他物体的中心距离
                    if self._check_center_distance(current_node, sampled_objects, max_center_distance):
                        # 放置物体
                        sampled_objects.append((current_node, self.node_positions[current_node], obj_type))
                        self._mark_area_occupied(current_node, occupied_areas, length, width)
                        placed = True
                        print(f"已放置 {obj_type} 在节点 {current_node}")
                        break
                
                # 如果当前节点不合适，随机游走到下一个节点
                neighbors = list(self.graph.neighbors(current_node))
                if not neighbors:
                    # 如果没有邻居，随机选择一个新起点
                    if sampled_objects:
                        # 从已放置物体附近开始
                        restart_from = random.choice(sampled_objects)[0]
                        current_node = restart_from
                    else:
                        current_node = random.choice(list(self.graph.nodes()))
                else:
                    # 随机选择下一个节点
                    current_node = random.choice(neighbors)
                
                # 概率性地从已放置物体重新开始
                if random.random() < 0.2 and sampled_objects:  # 20%概率
                    current_node = random.choice(sampled_objects)[0]
                    
                local_steps += 1
                steps += 1
            
            if not placed:
                print(f"警告: 无法为 {obj_type} ({length}x{width}) 找到合适位置")
        
        print(f"成功放置 {len(sampled_objects)}/{len(object_list)} 个物体，总步数: {steps}")
        return sampled_objects
    
    def _can_place_object_at(self, node, occupied_areas, length, width, min_distance):
        """检查节点是否可以放置物体"""
        if node not in self.node_positions:
            return False
            
        node_pos = np.array(self.node_positions[node])
        
        # 物体边界框
        obj_half_length = length / 2
        obj_half_width = width / 2
        obj_min_x = node_pos[0] - obj_half_length
        obj_min_y = node_pos[1] - obj_half_width
        obj_max_x = node_pos[0] + obj_half_length
        obj_max_y = node_pos[1] + obj_half_width
        
        # 检查是否与已占用区域重叠或过近
        for occ_node, occ_length, occ_width in occupied_areas:
            occ_pos = np.array(self.node_positions[occ_node])
            
            # 已占用物体的边界框
            occ_half_length = occ_length / 2
            occ_half_width = occ_width / 2
            occ_min_x = occ_pos[0] - occ_half_length
            occ_min_y = occ_pos[1] - occ_half_width
            occ_max_x = occ_pos[0] + occ_half_length
            occ_max_y = occ_pos[1] + occ_half_width
            
            # 检查是否重叠
            if not (obj_max_x < occ_min_x or obj_min_x > occ_max_x or 
                    obj_max_y < occ_min_y or obj_min_y > occ_max_y):
                return False
                
            # 检查边缘距离
            edge_dist_x = max(0, 
                             obj_min_x - occ_max_x if obj_min_x > occ_max_x else 
                             occ_min_x - obj_max_x if obj_max_x < occ_min_x else 0)
            edge_dist_y = max(0, 
                             obj_min_y - occ_max_y if obj_min_y > occ_max_y else 
                             occ_min_y - obj_max_y if obj_max_y < occ_min_y else 0)
            
            edge_dist = np.sqrt(edge_dist_x**2 + edge_dist_y**2)
            if edge_dist < min_distance:
                return False
        
        return True
    
    def _check_center_distance(self, node, sampled_objects, max_distance):
        """检查与其他物体中心的距离是否在范围内"""
        if not sampled_objects:
            return True
            
        node_pos = np.array(self.node_positions[node])
        
        # 检查至少与一个已放置物体的中心距离在范围内
        for sampled_node, _, _ in sampled_objects:
            sampled_pos = np.array(self.node_positions[sampled_node])
            dist = np.linalg.norm(node_pos - sampled_pos)
            if dist <= max_distance:
                return True
        
        return False
    
    def _mark_area_occupied(self, node, occupied_areas, length, width):
        """标记被物体占用的区域"""
        occupied_areas.append((node, length, width))
 
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
    
    def _check_clear_line_of_sight_direct(self, camera_pos, obj_pos, objects, target_idx):
        """直接检查相机与物体之间的视线是否畅通（使用位置而非节点）"""
        camera_pos = np.array(camera_pos)
        obj_pos = np.array(obj_pos)
        
        # 计算方向向量和距离
        direction = obj_pos - camera_pos
        distance = np.linalg.norm(direction)
        if distance == 0:
            return False
        direction = direction / distance
        
        # 检查其他物体是否遮挡视线
        for i, (_, pos, obj_type) in enumerate(objects):
            if i == target_idx:
                continue
                
            pos_array = np.array(pos)
            
            # 计算物体到相机-目标连线的投影距离
            camera_to_obj = pos_array - camera_pos
            proj_dist = np.dot(camera_to_obj, direction)
            
            # 只检查位于相机和目标之间的物体
            if 0 < proj_dist < distance:
                # 计算物体到连线的垂直距离
                perp_vector = camera_to_obj - proj_dist * direction
                perp_dist = np.linalg.norm(perp_vector)
                
                # 根据物体类型确定遮挡阈值
                if obj_type in ['car', 'truck'] and perp_dist < 150:
                    return False
                elif obj_type in ['human', 'animal'] and perp_dist < 50:
                    return False
                    
        return True

def visualize_cluster_cameras(objects, cameras, node_positions, object_types, output_file="cluster_cameras.png"):
    """可视化物体和集群相机位置（显示每个相机的视野）"""
    import math
    
    plt.figure(figsize=(16, 16))
    # 在绘图之前添加
    plt.gca().invert_yaxis()
    # 绘制背景点
    all_pos = np.array(list(node_positions.values()))
    plt.scatter(all_pos[:, 0], all_pos[:, 1], c='lightgray', s=1, alpha=0.2)
    
    # 为不同物体类型设置颜色
    colors = {
        'car': 'red',
        'human': 'blue',
        'bicycle': 'green',
        'truck': 'purple',
        'animal': 'orange'
    }
    
    # 物体计数
    type_counts = {}
    
    # 绘制物体
    for i, (node, pos, obj_type) in enumerate(objects):
        # 获取物体尺寸
        if obj_type in object_types:
            length, width = object_types[obj_type]
        else:
            length, width = 100, 100
            
        # 物体颜色
        if obj_type not in colors:
            colors[obj_type] = np.random.rand(3,)
        
        # 绘制物体
        plt.scatter(pos[0], pos[1], c=colors[obj_type], s=80, alpha=0.8)
        
        # 标注物体编号
        plt.annotate(f"{i}", (pos[0], pos[1]), fontsize=12)
        
        # 绘制物体轮廓
        rect = Rectangle(
            (pos[0] - length/2, pos[1] - width/2),
            length, width,
            linewidth=1.5, edgecolor=colors[obj_type], facecolor='none', alpha=0.7
        )
        plt.gca().add_patch(rect)
        
        # 更新计数
        if obj_type not in type_counts:
            type_counts[obj_type] = 0
        type_counts[obj_type] += 1
    
    # 为相机设置不同的颜色
    camera_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 绘制相机位置和视线
    for i, cam in enumerate(cameras):
        cam_pos = cam['position']
        cam_color = camera_colors[i % len(camera_colors)]
        
        # 绘制相机点
        plt.scatter(cam_pos[0], cam_pos[1], c=cam_color, marker='x', s=100)
        
        # 标注相机编号
        plt.annotate(f"C{i}", (cam_pos[0], cam_pos[1]), fontsize=12, color=cam_color)
        
        # 绘制相机到每个可见物体的连线
        for obj_idx in cam.get('visible_objects', []):
            if obj_idx < len(objects):
                target_pos = objects[obj_idx][1]
                plt.plot([cam_pos[0], target_pos[0]], [cam_pos[1], target_pos[1]], 
                        '--', color=cam_color, alpha=0.5, linewidth=1.5)
        
        # 绘制相机视场角
        angle = cam['rotation'][1] if 'rotation' in cam else 0  # 使用yaw值，如果没有则默认为0
        fov = 90  # 视场角度
        for fov_angle in [angle-fov/2, angle+fov/2]:
            fov_rad = -math.radians(fov_angle)
            fov_len = 400  # 视场线长度
            fov_x = cam_pos[0] + fov_len * math.cos(fov_rad)
            fov_y = cam_pos[1] + fov_len * math.sin(fov_rad)
            plt.plot([cam_pos[0], fov_x], [cam_pos[1], fov_y], 
                    '-', color=cam_color, alpha=0.4, linewidth=1.0)
    
    # 创建图例
    legend_elements = []
    for obj_type, count in type_counts.items():
        from matplotlib.lines import Line2D
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=colors[obj_type], markersize=10, 
                   label=f'{obj_type} ({count})')
        )
    
    # 添加相机图例
    legend_elements.append(
        Line2D([0], [0], marker='x', color='black', 
               markerfacecolor='black', markersize=10, 
               label=f'Camera ({len(cameras)})')
    )
    
    plt.title(f"Objects and Cluster Cameras - {len(objects)} objects, {len(cameras)} cameras")
    plt.legend(handles=legend_elements)
    plt.axis('equal')
    plt.grid(alpha=0.3)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存可视化到 {output_file}")


if __name__ == "__main__":
    # 初始化采样器
    env = "SuburbNeighborhood_Day"
    sampler = GraphBasedSampler(f"./points_graph/{env}/environment_graph.gpickle")
    
    # 定义物体类型及尺寸
    object_types = {
        'car': (400, 200),       # 汽车: 4m×2m
        'human': (50, 50),       # 人: 0.5m×0.5m
        'bicycle': (180, 60),    # 自行车: 1.8m×0.6m
        'truck': (600, 250),     # 卡车: 6m×2.5m
        'animal': (100, 50)      # 动物: 1m×0.5mcd
    }
    
    # 采样物体位置并为整个集群采样相机位置
    objects, cameras = sampler.sample_objects_with_cluster_cameras(
        object_types=object_types,
        num_objects=7,
        min_edge_distance=100,
        max_center_distance=1500,
        camera_count=3  # 目标是3个相机点，但保证至少有2个
    )
    
    # 可视化结果前检查相机数量
    print(f"最终获得 {len(cameras)} 个相机点")
    print(f"物体位置: {objects}\n",f"相机位置: {cameras}")
    # 可视化结果
    visualize_cluster_cameras(
        objects, cameras, 
        sampler.node_positions, object_types, 
        "cluster_cameras.png"
    )