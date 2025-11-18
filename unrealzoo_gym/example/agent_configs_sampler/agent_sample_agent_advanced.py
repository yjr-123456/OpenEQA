import numpy as np
from .agent_sample_agent import AgentBasedSampler
import cv2
import os
import svgwrite

class AgentBasedSamplerboost(AgentBasedSampler):
    """增强版采样器，支持自适应几何约束"""
    
    def __init__(self, graph_pickle_file, model, config_path="model_config.json"):
        """初始化增强型采样器
        
        Args:
            graph_pickle_file: 图的pickle文件路径
            model: VLM模型名称
            config_path: 配置文件路径
        """
        # 调用父类初始化
        super().__init__(graph_pickle_file, model, config_path)
        # 可以在这里添加新的属性如果需要
        self.adaptive_mask_params = {}
        self.cam_relative_height = 1200  # 默认相机高度（厘米）

    def save_scene_as_svg(self, save_path, background_image_path, W, H):
        """
        将一张光栅图直接嵌入到一个 SVG 文件中，不添加任何额外元素。

        Args:
            save_path (str): SVG 文件保存路径。
            background_image_path (str): 作为背景的 PNG/JPG 图像路径。
            W (int): 图像宽度。
            H (int): 图像高度。
        """
        try:
            dwg = svgwrite.Drawing(save_path, profile='tiny', size=(W, H))
            dwg.add(dwg.image(href=background_image_path, insert=(0, 0), size=(W, H)))
            dwg.save()
            print(f"[Debug] 已保存 SVG 包装: {save_path}")
        except Exception as e:
            print(f"❌ 保存 SVG 失败: {e}")



    def calculate_collision_radius(self, length, width, yaw_rad):
        """
        基于物体尺寸和旋转，计算该物体的碰撞半径（外接圆）。
        
        Args:
            length (float): 物体长度（沿X轴）
            width (float): 物体宽度（沿Y轴）
            yaw_rad (float): 旋转角度（弧度）
        
        Returns:
            collision_radius (float): 最小外接圆半径（厘米）
        """
        # 计算物体四个角到中心的距离
        corners = np.array([
            [length / 2, width / 2],
            [length / 2, -width / 2],
            [-length / 2, width / 2],
            [-length / 2, -width / 2]
        ])
        
        # 不管旋转角度如何，外接圆半径都是对角线长度的一半
        # 这是因为旋转不改变点到中心的距离
        diagonal = np.sqrt(length**2 + width**2)
        collision_radius = diagonal / 2.0
        
        return collision_radius
    
    
    def calculate_adaptive_erosion_radius(self, obj_length, obj_width, yaw_rad, 
                                        K, cam_height, safety_margin_cm=50):
        """
        基于物体尺寸、旋转和相机参数，计算在图像空间中的自适应腐蚀半径。
        
        Args:
            obj_length (float): 物体长度（厘米）
            obj_width (float): 物体宽度（厘米）
            yaw_rad (float): 旋转角度（弧度，UnrealCV坐标系）
            K (np.ndarray): 相机内参矩阵
            cam_height (float): 俯视相机高度（厘米）
            safety_margin_cm (float): 额外的安全边距（厘米）
        
        Returns:
            erosion_radius_px (int): 在图像空间中的腐蚀半径（像素）
        """
        # 1. 计算物体的碰撞半径（外接圆）
        collision_radius_cm = self.calculate_collision_radius(obj_length, obj_width, yaw_rad)
        
        # 2. 加上安全边距
        total_radius_cm = collision_radius_cm + safety_margin_cm
        
        # 3. 将厘米转换为图像像素
        # 在俯视图中，焦距为 f，高度为 h
        # 物体大小 d_cm 在图像中的像素大小为：d_px = f * d_cm / h
        focal_length_px = K[0, 0]
        pixel_per_cm = focal_length_px / cam_height
        erosion_radius_px = int(total_radius_cm * pixel_per_cm)
        
        return erosion_radius_px
         
    def apply_occupied_area_erosion(self, placeable_mask, occupied_area, K, cam_height, 
                                   safety_margin_cm=50):
        """
        针对单个已放置的物体，对可放置掩码进行自适应腐蚀。
        
        Args:
            placeable_mask (np.ndarray): 当前的可放置掩码
            occupied_area (dict): 已放置物体的占据信息
                                 {'center_pos': [x, y, z], 'length': int, 'width': int, 
                                  'yaw': float, 'corners': [...]}
            K (np.ndarray): 相机内参矩阵
            cam_height (float): 相机高度（厘米）
            safety_margin_cm (float): 额外安全边距（厘米）
        
        Returns:
            updated_mask (np.ndarray): 更新后的可放置掩码
        """
        import cv2
        
        obj_length = occupied_area['length']
        obj_width = occupied_area['width']
        yaw_deg = occupied_area['yaw']
        yaw_rad = np.radians(yaw_deg)
        center_pos = occupied_area['center_pos']
        
        # 1. 计算这个物体特定的腐蚀半径
        erosion_radius_px = self.calculate_adaptive_erosion_radius(
            obj_length, obj_width, yaw_rad,
            K, cam_height,
            safety_margin_cm=safety_margin_cm
        )
        
        print(f"[OccupiedErosion] 物体在 {center_pos[:2]} 处，尺寸: {obj_length}x{obj_width}cm, 旋转: {yaw_deg:.1f}°")
        print(f"[OccupiedErosion] 应用腐蚀半径: {erosion_radius_px} 像素")
        
        # 2. 创建掩码来标记这个物体及其周围的安全区
        H, W = placeable_mask.shape
        occupied_mask = np.zeros((H, W), dtype=np.uint8)
        
        # 将物体的四个角投影到图像上
        world_corners = occupied_area['corners']  # shape: (4, 2)
        
        for corner in world_corners:
            corner_3d = np.array([corner[0], corner[1], center_pos[2]])
            u, v, depth = self.world_2_image_unreal(corner_3d, self.cam_pose, K)
            
            if depth > 0 and 0 <= u < W and 0 <= v < H:
                occupied_mask[int(v), int(u)] = 255
        
        # 3. 对物体占据的区域进行膨胀（以标记占据区）
        dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * erosion_radius_px + 1, 2 * erosion_radius_px + 1)
        )
        
        occupied_expanded = cv2.dilate(occupied_mask, dilation_kernel, iterations=1)
        
        # 4. 从可放置掩码中移除已占据和已腐蚀的区域
        # occupied_expanded 中的像素 > 0 表示不可放置
        updated_mask = placeable_mask.copy()
        updated_mask[occupied_expanded > 0] = 0
        
        return updated_mask

    def generate_base_placeable_mask(self, env, cam_id, 
                                normal_variance_threshold=0.05,
                                slope_threshold=0.866,
                                gaussian_kernel_size=5,
                                gaussian_sigma=1.0,
                                W=None, H=None,save_dir="./test_results/"):
        """
        生成基础可放置掩码（向量化优化版本）
        """
        print("[BasePreprocess] 正在生成基础可放置掩码...")

        def _save_normal_map(normal_map, filename_prefix):
            """
            辅助函数，对法线图进行对比度增强并保存为单个彩色图像。
            """
            # 1. 确保我们有 uint8 格式的图像用于处理
            if normal_map.dtype == np.uint8:
                normal_uint8 = normal_map
            else: 
                # 从 float 格式 [-1, 1] 转换回 uint8 [0, 255]
                normal_uint8 = (normal_map * 127.0 + 128.0).clip(0, 255).astype(np.uint8)

            # 2. 创建一个新的空图像用于存放增强后的结果
            enhanced_color = np.zeros_like(normal_uint8)
            
            # 3. 对 R, G, B 每个通道独立进行对比度拉伸
            for i in range(3):
                # cv2.normalize 会找到当前通道的 min/max 值，并将其拉伸到 0-255
                normalized_channel = cv2.normalize(normal_uint8[:,:,i], None, 0, 255, cv2.NORM_MINMAX)
                enhanced_color[:,:,i] = normalized_channel
            
            # 4. 转换颜色空间并保存
            enhanced_color_bgr = cv2.cvtColor(enhanced_color, cv2.COLOR_RGB2BGR)
            filepath = os.path.join(save_dir, f"{filename_prefix}_enhanced.png")
            cv2.imwrite(filepath, enhanced_color_bgr)
            print(f"[Debug] 已保存对比度增强法线图: {filepath}")



        # 1. 获取法线图
        normal_bgr = env.unrealcv.read_image(cam_id, 'normal')
        normal_rgb = cv2.cvtColor(normal_bgr, cv2.COLOR_BGR2RGB)
        _save_normal_map(normal_rgb, "debug_normal_01_raw.png")

        if W is None or H is None:
            H, W = normal_rgb.shape[:2]
        
        print(f"[BasePreprocess] 法线图形状: {normal_rgb.shape}")
        
        # 2. 法线解码 + 归一化
        normal_float = (normal_rgb.astype(np.float32) - 128.0) / 127.0
        normal_length = np.linalg.norm(normal_float, axis=2, keepdims=True)
        normal_normalized = np.divide(
            normal_float, 
            normal_length, 
            out=np.zeros_like(normal_float),
            where=normal_length > 0.1
        )
        _save_normal_map(normal_normalized, "debug_normal_02_normalized.png")
        # 3. 高斯平滑
        print("[BasePreprocess] 应用高斯平滑...")
        normal_smoothed = cv2.GaussianBlur(
            normal_normalized, 
            (gaussian_kernel_size, gaussian_kernel_size), 
            gaussian_sigma
        )
        _save_normal_map(normal_smoothed, "debug_normal_03_smoothed.png")
        # 4. 重新归一化
        normal_smoothed_length = np.linalg.norm(normal_smoothed, axis=2, keepdims=True)
        final_normal_map = np.divide(
            normal_smoothed,
            normal_smoothed_length,
            out=np.zeros_like(normal_smoothed),
            where=normal_smoothed_length > 0.1
        )
        _save_normal_map(final_normal_map, "debug_normal_04_final.png")
        # ========================================
        # === 🚀 关键优化：向量化计算平滑度 ===
        # ========================================
        print("[BasePreprocess] 计算平滑度掩码（向量化版本）...")
        
        kernel_size = 5
        pad_size = kernel_size // 2
        
        # 使用 cv2.boxFilter 计算局部均值（超快！）
        mean_normal = cv2.boxFilter(
            final_normal_map, 
            ddepth=-1, 
            ksize=(kernel_size, kernel_size),
            normalize=True,
            borderType=cv2.BORDER_REPLICATE
        )
        
        # 归一化均值法线
        mean_normal_length = np.linalg.norm(mean_normal, axis=2, keepdims=True)
        mean_normal_normalized = np.divide(
            mean_normal,
            mean_normal_length,
            out=np.zeros_like(mean_normal),
            where=mean_normal_length > 1e-6
        )
        
        # 使用 uniform_filter 计算局部方差（向量化）
        from scipy.ndimage import uniform_filter
        
        # 计算每个像素与其局部均值法线的点积
        dot_product = np.sum(final_normal_map * mean_normal_normalized, axis=2)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # 角度
        angles = np.arccos(dot_product)
        
        # 计算局部方差（使用 Var(X) = E[X²] - E[X]² ）
        angles_squared = angles ** 2
        local_mean_sq = uniform_filter(angles_squared, size=kernel_size, mode='nearest')
        local_mean = uniform_filter(angles, size=kernel_size, mode='nearest')
        smooth_map = local_mean_sq - local_mean ** 2
        
        # 处理无效区域
        smooth_map[mean_normal_length[:, :, 0] < 1e-6] = 255
        
        smooth_mask = (smooth_map < normal_variance_threshold).astype(np.uint8) * 255
        print(f"[BasePreprocess] 平滑区域占比: {(smooth_map < normal_variance_threshold).sum() / (H * W) * 100:.2f}%")
        
        # ========================================
        
        # 6. 计算坡度掩码（已经是向量化的）
        print("[BasePreprocess] 计算坡度掩码...")
        vertical_direction = np.array([0.0, 0.0, 1.0])
        dot_product = np.dot(final_normal_map, vertical_direction)
        slope_mask = (dot_product > slope_threshold).astype(np.uint8) * 255
        print(f"[BasePreprocess] 不陡峭区域占比: {(dot_product > slope_threshold).sum() / (H * W) * 100:.2f}%")
        
        # 7. 合并基础掩码
        base_mask = cv2.bitwise_and(smooth_mask, slope_mask)
        print(f"[BasePreprocess] 基础掩码完成，可放置区域: {(base_mask > 0).sum() / (H * W) * 100:.2f}%")
        
        self.visualize_mask(smooth_mask, "debug_mask_01_smooth.png", save_dir)
        self.visualize_mask(slope_mask, "debug_mask_02_slope.png", save_dir)
        self.visualize_mask(base_mask, "debug_mask_03_base_placeable.png", save_dir)
        
        # # 可选：叠加在原始俯视图上
        # self.overlay_mask_on_image(obs_rgb, base_mask, alpha=0.4, 
        #                            filename="debug_mask_04_overlay_on_topview.png", 
        #                            save_dir=save_dir)  


        return base_mask, final_normal_map, (W, H)

    def apply_adaptive_erosion_to_mask(self, base_mask, next_object_info, 
                                   safety_margin_cm=50):
        """
        对基础掩码应用物体特定的自适应腐蚀（快速版本）
        
        Args:
            base_mask: 基础可放置掩码
            next_object_info: 待放置物体的信息
            safety_margin_cm: 安全边距
        
        Returns:
            adaptive_mask: 应用腐蚀后的掩码
            erosion_radius: 腐蚀半径（像素）
        """
        obj_length = next_object_info.get('length', 100)
        obj_width = next_object_info.get('width', 100)
        rotation = next_object_info.get('rotation', [0, 0, 0])
        yaw_deg = rotation[1] if len(rotation) > 1 else 0
        yaw_rad = np.radians(yaw_deg)
        

        
        # 计算腐蚀半径
        erosion_radius_px = self.calculate_adaptive_erosion_radius(
            obj_length, obj_width, yaw_rad, self.K, self.cam_relative_height,
            safety_margin_cm=safety_margin_cm
        )
        
        # 应用腐蚀
        erosion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * erosion_radius_px + 1, 2 * erosion_radius_px + 1)
        )
        
        adaptive_mask = cv2.erode(base_mask, erosion_kernel, iterations=1)
        
        return adaptive_mask, erosion_radius_px

    def visualize_mask(self, mask, filename, save_dir="./test_results/", colormap=cv2.COLORMAP_JET):
        """
        可视化掩码为彩色图像并保存。
        
        Args:
            mask (np.ndarray): 二值或灰度掩码 (H, W)
            filename (str): 保存的文件名
            save_dir (str): 保存目录
            colormap: OpenCV 的色彩映射方案
        
        Returns:
            colored_mask (np.ndarray): 彩色化后的掩码
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 确保掩码是 uint8 格式
        if mask.dtype != np.uint8:
            mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)
        else:
            mask_uint8 = mask
        
        # 应用色彩映射
        colored_mask = cv2.applyColorMap(mask_uint8, colormap)
        
        # 保存
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, colored_mask)
        print(f"[Visualization] 已保存掩码可视化: {filepath}")
        
        return colored_mask

    def overlay_mask_on_image(self, image, mask, alpha=0.5, filename=None, save_dir="./test_results/"):
        """
        将掩码叠加在原始图像上，用于可视化效果。
        
        Args:
            image (np.ndarray): 原始RGB图像 (H, W, 3)
            mask (np.ndarray): 二值掩码 (H, W)
            alpha (float): 掩码的透明度 (0-1)
            filename (str): 保存文件名 (可选)
            save_dir (str): 保存目录
        
        Returns:
            overlayed (np.ndarray): 叠加后的RGB图像
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 确保数据类型一致
        image = image.astype(np.float32)
        mask = mask.astype(np.float32) / 255.0 if mask.max() > 1.0 else mask.astype(np.float32)
        
        # 创建彩色掩码（绿色表示可放置区域）
        colored_mask = np.zeros_like(image)
        colored_mask[:, :, 1] = mask * 255  # 绿色通道
        
        # 叠加
        overlayed = (image * (1 - alpha) + colored_mask * alpha).astype(np.uint8)
        
        if filename:
            filepath = os.path.join(save_dir, filename)
            overlayed_bgr = cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, overlayed_bgr)
            print(f"[Visualization] 已保存叠加图像: {filepath}")
        
        return overlayed

    def run_sampling_experiment(self, env, agent_configs, experiment_config, cam_id=0, 
                               cam_count=3, vehicle_zones=None, height=800, **kwargs):
        """
        优化版本：预处理基础掩码 + 循环内自适应腐蚀
        """
        # 1-4. 保持原有逻辑...
        save_dir = kwargs.get('save_dir', './test_results/')
        os.makedirs(save_dir, exist_ok=True)
        object_list, all_objects_are_small, has_car = self.sort_objects(agent_configs)
        vehicle_zone_nodes = self.filter_car_zones(vehicle_zones)
        agent_sampling_center_pos, center_node = self.sample_center_point(vehicle_zone_nodes, has_car, all_objects_are_small)
        
        self.ground_z = agent_sampling_center_pos[2]
        orginal_cam_pose = env.unrealcv.get_cam_location(cam_id) + env.unrealcv.get_cam_rotation(cam_id)
        
        if agent_sampling_center_pos is not None:
            env.unrealcv.set_cam_location(cam_id, np.append(agent_sampling_center_pos[:2], agent_sampling_center_pos[2] + height))
            env.unrealcv.set_cam_rotation(cam_id, [-90, 0, 0])
        self.cam_relative_height = height
        obs_bgr = env.unrealcv.read_image(cam_id, 'lit')
        obs_rgb = cv2.cvtColor(obs_bgr, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{save_dir}/debug_topdown_view.png", obs_bgr)
        # self.save_scene_as_svg(
        #     save_path=f"{save_dir}/debug_sampling_step_{i+1}.svg",
        #     background_image_path=png_path,
        #     W=self.W, H=self.H
        # )
        cam_location = env.unrealcv.get_cam_location(cam_id)
        cam_rotation = env.unrealcv.get_cam_rotation(cam_id)
        self.cam_pose = cam_location + cam_rotation
        self.W, self.H = obs_rgb.shape[1], obs_rgb.shape[0]
        self.fov_deg = float(env.unrealcv.get_cam_fov(cam_id))
        self.K = self.get_camera_matrix_unreal(self.W, self.H, self.fov_deg)
        
        img_points, valid_mask, depths = self.project_points_to_image_unreal(
            self.node_list, self.cam_pose, self.W, self.H, self.fov_deg
        )
        
        all_valid_points_dict = {}
        for i, (node, node_id, valid) in enumerate(zip(self.node_list, self.node_id_list, valid_mask)):
            if valid:
                all_valid_points_dict[tuple(node)] = {'index': i, 'node': node_id}

        result_img = self.visualize_projected_points_unreal_with_next_object(
                obs_rgb, img_points, valid_mask , depths, self.W, self.H)
        result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{save_dir}/debug_initial_projection.png", result_img_bgr)
        
        # ========================================
        # === 🚀 关键优化：预处理基础掩码（只执行一次） ===
        # ========================================
        print("\n" + "="*60)
        print("🚀 开始预处理基础可放置掩码（执行1次）...")
        print("="*60)
        
        base_placeable_mask, final_normal_map, (W, H) = self.generate_base_placeable_mask(
            env=env,
            cam_id=cam_id,
            normal_variance_threshold=kwargs.get('normal_variance_threshold', 0.05),
            slope_threshold=kwargs.get('slope_threshold', 0.866),
            W=self.W,
            H=self.H,
            save_dir=save_dir,
            gaussian_kernel_size=kwargs.get('gaussian_kernel_size', 5),
            gaussian_sigma=kwargs.get('gaussian_sigma', 1.0)
        )
        
        print("✅ 基础掩码预处理完成！\n")
        
        # === 主采样循环 ===
        sampled_objects = []
        occupied_areas = []
        pending_objects = []
        
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
            
            # ========================================
            # === 🚀 优化：快速应用自适应腐蚀 ===
            # ========================================
            print(f"\n[Sampling] 物体 {i+1}/{len(object_list)} ({name}): 应用自适应腐蚀...")

            # 从基础掩码开始
            adaptive_mask = base_placeable_mask.copy()
          
            # 应用物体特定的腐蚀
            adaptive_mask, erosion_radius = self.apply_adaptive_erosion_to_mask(
                adaptive_mask,
                current_obj_details,
                safety_margin_cm=kwargs.get('safety_margin_cm', 50)
            )
            self.visualize_mask(adaptive_mask, f"debug_step_{i+1}_a_base_mask.png", save_dir)

            print(f"[Sampling] 物体腐蚀半径: {erosion_radius}px")
            
            # 对所有已放置的物体应用腐蚀
            for j, occ_area in enumerate(occupied_areas):
                adaptive_mask = self.apply_occupied_area_erosion(
                    adaptive_mask, occ_area, self.K, self.cam_relative_height,
                    safety_margin_cm=kwargs.get('safety_margin_cm', 50)
                )
            
            print(f"[Sampling] 最终可放置区域: {(adaptive_mask > 0).sum() / (H * W) * 100:.2f}%")
            self.visualize_mask(adaptive_mask, f"debug_step_{i+1}_d_final_adaptive_mask.png", save_dir)
            
            # 叠加在俯视图上查看效果
            self.overlay_mask_on_image(obs_rgb, adaptive_mask, alpha=0.5,
                                       filename=f"debug_step_{i+1}_e_final_adaptive_overlay.png",
                                       save_dir=save_dir)
                    
            
            # 使用自适应掩码筛选有效点
            geometry_filtered_mask = self.filter_valid_points_by_placeable_mask(
                img_points, valid_mask, adaptive_mask, margin_pixels=10
            )
            
            updated_valid_points_dict = {}  # 世界坐标 -> {index, node}
            node_to_world_map = {}          # 图节点 -> 世界坐标
            
            for j, (node, node_id, geom_valid) in enumerate(zip(
                self.node_list, self.node_id_list, geometry_filtered_mask
            )):
                if geom_valid:
                    world_pos = tuple(node)  # 世界坐标 (x, y, z)
                    updated_valid_points_dict[world_pos] = {'index': j, 'node': node_id}
                    node_to_world_map[node_id] = world_pos  # 反向映射
            
            print(f"[Sampling] 物体 {name}: 初始 {valid_mask.sum()} -> 几何筛选后 {geometry_filtered_mask.sum()}")
            # geom_filtered_mask_uint8 = geometry_filtered_mask.astype(np.uint8) * 255
            # self.visualize_mask(geom_filtered_mask_uint8, f"debug_step_{i+1}_f_geometry_filtered_mask.png", save_dir)
            
            # 可视化
            result_img = self.visualize_projected_points_unreal_with_next_object(
                obs_rgb, img_points, geometry_filtered_mask, depths, self.W, self.H,
                occupied_areas, current_obj_details
            )
            result_img_no_next_obj = self.visualize_projected_points_unreal_with_next_object(
                obs_rgb, img_points, geometry_filtered_mask, depths, self.W, self.H,
                occupied_areas
            )
            result_img_no_next_obj_bgr = cv2.cvtColor(result_img_no_next_obj, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{save_dir}/debug_sampling_step_{i+1}_no_next_obj.png", result_img_no_next_obj_bgr)
            print(f"[Debug] 已保存可视化: {save_dir}/debug_sampling_step_{i+1}.png")

            # 根据模式选择点
            use_fast_mode = kwargs.get('fast_test_mode', False)
            
            if use_fast_mode:
                # 快速模式：随机选点（从世界坐标中选）
                if len(updated_valid_points_dict) == 0:
                    print(f"⚠️  警告：物体 {name} 没有可用点，跳过。")
                    current_node_to_try = None
                else:
                    import random
                    world_pos = random.choice(list(updated_valid_points_dict.keys()))
                    node_id = updated_valid_points_dict[world_pos]['node']
                    print(f"✅ [FastMode] 随机选择点: Node {node_id} at {world_pos}")
                    current_node_to_try = node_id  # ← 使用世界坐标
            else:
                # ===== 🚀 标准模式：VLM选点（需要转换） =====
                selected_node_id = self.sample_object_points(
                    result_img, name, length, width, updated_valid_points_dict, 
                    experiment_config
                )
                
                if selected_node_id is None:
                    print(f"⚠️  警告：VLM未返回有效节点，跳过物体 {name}")
                    current_node_to_try = None
                elif selected_node_id in node_to_world_map:
                    # 关键：将图节点转换为世界坐标
                    current_node_to_try = selected_node_id
                    current_node_coord = node_to_world_map[selected_node_id]
                    print(f"✅ [VLM] 选择点: Node {selected_node_id} -> World {current_node_coord}")
                else:
                    print(f"⚠️  错误：VLM返回的节点 {selected_node_id} 不在有效点集合中")
                    current_node_to_try = None

            if current_node_to_try is None:
                print(f"警告：为物体 {name} 采样位置失败，跳过此物体。")
                continue
            
            position = self.node_positions[current_node_to_try]
            
            object_dict = {
                'node': current_node_to_try, 'position': position, 
                'rotation': current_obj_details['rotation'],
                'agent_type': agent_type, 'type': type_val, 'name': name, 
                'app_id': app_id, 'animation': animation, 
                'feature_caption': feature_caption, 'dimensions': (length, width)
            }
            sampled_objects.append(object_dict)
            
            # 更新有效点掩码
            valid_mask = self._mark_area_occupied(
                current_node_to_try, occupied_areas, valid_mask, length, width, yaw
            )
            
            # 可视化：更新后的有效点掩码（用于下一次迭代）
            # valid_mask_uint8 = valid_mask.astype(np.uint8) * 255
            # self.visualize_mask(valid_mask_uint8, f"debug_step_{i+1}_i_updated_valid_mask_for_next.png", save_dir)
            
            all_valid_points_dict = {}
            for j, (node, node_id, valid) in enumerate(zip(self.node_list, self.node_id_list, valid_mask)):
                if valid:
                    all_valid_points_dict[tuple(node)] = {'index': j, 'node': node_id}
        
        # 5-10. 相机采样和返回（保持不变）...
        cameras = self._sample_external_cameras(
            objects=sampled_objects,
            camera_count=cam_count,
            ring_inner_radius_offset=kwargs.get('ring_inner_radius_offset', 200),
            ring_outer_radius_offset=kwargs.get('ring_outer_radius_offset', 800),
            min_angle_separation_deg=kwargs.get('min_angle_separation_deg', 30),
            min_cam_to_agent_dist=kwargs.get('min_cam_to_agent_dist', 150)
        )
        
        result_img_with_cams = self.visualize_projected_points_unreal_with_cameras(
            obs_rgb, img_points, valid_mask, depths, self.W, self.H,
            occupied_areas, cameras
        )
        result_img_rgb = cv2.cvtColor(result_img_with_cams, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{save_dir}/debug_sampling_step_with_cameras_optimized.png", result_img_rgb)
        
        camera_id_list = self.sample_camera_points(result_img_rgb)
        
        selected_cameras = []
        if camera_id_list:
            for cam_info in cameras:
                if cam_info["id"] in camera_id_list:
                    selected_cameras.append(cam_info)
        
        updated_configs, camera_configs = self.format_transform(agent_configs, sampled_objects, selected_cameras)
        center_pos_to_return = agent_sampling_center_pos.tolist() if isinstance(agent_sampling_center_pos, np.ndarray) else agent_sampling_center_pos
        all_distances = [np.linalg.norm(np.array(obj['position']) - agent_sampling_center_pos) for obj in sampled_objects]
        agent_sampling_radius = max(all_distances) + 200 if all_distances else 200
        
        env.unrealcv.set_cam_location(cam_id, orginal_cam_pose[:3])
        env.unrealcv.set_cam_rotation(cam_id, orginal_cam_pose[3:])
        
        return {
            "env": env,
            'agent_configs': updated_configs,
            'camera_configs': camera_configs,
            'sampling_center': center_pos_to_return,
            'sampling_radius': agent_sampling_radius
        }

    def sample_agent_positions(self, env, agent_configs, cam_id=0, cam_count=3, 
                                vehicle_zones=None, all_max_distance=None,
                                ring_inner_radius_offset=300,
                                ring_outer_radius_offset=500,
                                min_angle_separation_deg=35,
                                min_cam_to_agent_dist=200,
                                height=800, **kwargs):
            """
            兼容原方法的包装器。自动调用新的自适应采样方法。
            
            Args:
                env: 环境实例
                agent_configs: 智能体配置
                cam_id: 相机ID
                cam_count: 相机数量
                vehicle_zones: 车辆区域
                all_max_distance: 最大距离
                ring_inner_radius_offset: 环内半径偏移
                ring_outer_radius_offset: 环外半径偏移
                min_angle_separation_deg: 最小角度分离
                min_cam_to_agent_dist: 最小相机到智能体距离
                height: 相机高度
                **kwargs: 其他参数
            
            Returns:
                dict: 采样结果
            """
            print("[兼容性包装器] 调用自适应采样方法...")
            
            # 直接调用新方法
            result = self.run_sampling_experiment(
                env=env,
                agent_configs=agent_configs,
                experiment_config={'type': 'data_generation'},
                cam_id=cam_id,
                cam_count=cam_count,
                vehicle_zones=vehicle_zones,
                height=height,
                ring_inner_radius_offset=ring_inner_radius_offset,
                ring_outer_radius_offset=ring_outer_radius_offset,
                min_angle_separation_deg=min_angle_separation_deg,
                min_cam_to_agent_dist=min_cam_to_agent_dist,
                **kwargs
            )
            
            return result


    
    def filter_valid_points_by_placeable_mask(self, img_points, valid_mask, placeable_mask, 
                                             margin_pixels=5):
        """
        使用可放置掩码进一步筛选有效点。
        """
        H, W = placeable_mask.shape
        refined_mask = valid_mask.copy()
        
        for idx, (point, is_valid) in enumerate(zip(img_points, valid_mask)):
            if not is_valid:
                continue
            
            u, v = int(point[0]), int(point[1])
            
            if (margin_pixels <= u < W - margin_pixels and 
                margin_pixels <= v < H - margin_pixels):
                if placeable_mask[v, u] == 0:
                    refined_mask[idx] = False
            else:
                refined_mask[idx] = False
        
        print(f"[GeometryFilter] 几何筛选后，有效点数: {refined_mask.sum()} / {len(refined_mask)}")
        return refined_mask