import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import os
def depth_to_pointcloud(depth_image, camera_intrinsics=None, rgb_image=None, depth_scale=1.0, depth_threshold=10000.0, use_inverse=True):
    """
    将深度图转换为点云
    
    Args:
        depth_image: 深度图 (H, W) numpy数组
        camera_intrinsics: 相机内参 dict {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
        rgb_image: RGB图像 (H, W, 3) numpy数组，可选
        depth_scale: 深度缩放因子，默认1.0
        depth_threshold: 深度阈值，超过此值的深度会被处理
        use_inverse: 是否对超过阈值的深度使用倒数处理
    
    Returns:
        points: 点云坐标 (N, 3)
        colors: 点云颜色 (N, 3)，如果提供了RGB图像
    """
    height, width = depth_image.shape
    
    # 默认相机内参（如果未提供）
    if camera_intrinsics is None:
        camera_intrinsics = {
            'fx': width * 0.8,  # 近似焦距
            'fy': height * 0.8,
            'cx': width / 2.0,   # 图像中心
            'cy': height / 2.0
        }
    
    fx, fy = camera_intrinsics['fx'], camera_intrinsics['fy']
    cx, cy = camera_intrinsics['cx'], camera_intrinsics['cy']
    
    # 创建像素坐标网格
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # 处理深度值
    processed_depth = depth_image.copy()
    
    if use_inverse:
        # 对超过阈值的深度使用倒数处理
        over_threshold_mask = processed_depth > depth_threshold
        print(f"处理深度阈值: {depth_threshold}")
        print(f"超过阈值的像素数: {np.sum(over_threshold_mask)}/{processed_depth.size}")
        
        if np.sum(over_threshold_mask) > 0:
            # 使用倒数，避免除零
            processed_depth[over_threshold_mask] = 1.0 / (processed_depth[over_threshold_mask] + 1e-8)
            print(f"处理后深度范围: [{processed_depth.min():.6f}, {processed_depth.max():.6f}]")
    
    # 获取有效深度的像素
    valid_depth = processed_depth > 0
    u_valid = u[valid_depth]
    v_valid = v[valid_depth]
    depth_valid = processed_depth[valid_depth] * depth_scale  # 这里是修改的第42行
    
    # 转换为3D坐标
    x = (u_valid - cx) * depth_valid / fx
    y = (v_valid - cy) * depth_valid / fy
    z = depth_valid
    
    points = np.column_stack((x, y, z))
    
    # 如果提供了RGB图像，提取颜色
    colors = None
    if rgb_image is not None:
        rgb_valid = rgb_image[valid_depth]
        colors = rgb_valid / 255.0  # 归一化到[0,1]
    
    return points, colors

def visualize_pointcloud_matplotlib(points, colors=None, title="Point Cloud"):
    """
    使用matplotlib可视化点云
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if colors is not None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=colors, s=1, alpha=0.6)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=points[:, 2], cmap='viridis', s=1, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Depth)')
    ax.set_title(title)
    
    # 设置相等的坐标比例
    max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                         points[:, 1].max() - points[:, 1].min(),
                         points[:, 2].max() - points[:, 2].min()]).max() / 2.0
    
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()

def visualize_pointcloud_open3d(points, colors=None, title="Point Cloud"):
    """
    使用Open3D可视化点云（更好的交互性）
    """
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # 使用深度值作为颜色
        depth_colors = plt.cm.viridis(points[:, 2] / points[:, 2].max())[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(depth_colors)
    
    # 可视化
    o3d.visualization.draw_geometries([pcd], 
                                    window_name=title,
                                    width=800, height=600)

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def reconstruct_pointcloud_from_observation(obs_rgb, obs_depth, save_dir=None, env_name="unknown"):
    """
    从环境观察数据重建点云
    """
    # 提取RGB和深度数据
    # obs_rgb = states[0][..., :3].squeeze()
    # obs_depth = states[0][..., -1].squeeze()
    
    print(f"=== 点云重建 ===")
    print(f"RGB shape: {obs_rgb.shape}")
    print(f"Depth shape: {obs_depth.shape}")
    print(f"Depth range: [{obs_depth.min():.3f}, {obs_depth.max():.3f}]")
    
    # 转换RGB格式
    obs_rgb_converted = cv2.cvtColor(obs_rgb, cv2.COLOR_BGR2RGB)
    
    # 估算相机内参（基于图像尺寸）
    height, width = obs_depth.shape
    camera_intrinsics = {
        'fx': 256.0,   # 焦距（像素）
        'fy': 256.0,
        'cx': 256.0,   # 主点（像素）
        'cy': 256.0
    }

    # 调用点云重建
    points, colors = depth_to_pointcloud(
        depth_image=obs_depth,
        camera_intrinsics=camera_intrinsics,
        rgb_image=obs_rgb_converted,
        depth_scale=1
    )
    
    print(f"重建的点云: {len(points)} 个点")
    
    # 保存点云数据
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存为numpy格式
        np.save(os.path.join(save_dir, f"{env_name}_pointcloud_xyz.npy"), points)
        if colors is not None:
            np.save(os.path.join(save_dir, f"{env_name}_pointcloud_colors.npy"), colors)
        
        # 保存为PLY格式（可用MeshLab等软件打开）
        save_pointcloud_ply(points, colors, 
                           os.path.join(save_dir, f"{env_name}_pointcloud.ply"))
        
        print(f"点云数据已保存到: {save_dir}")
    
    return points, colors

def save_pointcloud_ply(points, colors=None, filename="pointcloud.ply"):
    """
    保存点云为PLY格式
    """
    with open(filename, 'w') as f:
        # PLY文件头
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        
        f.write("end_header\n")
        
        # 写入点云数据
        for i in range(len(points)):
            x, y, z = points[i]
            if colors is not None:
                r, g, b = (colors[i] * 255).astype(int)
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    
    print(f"PLY文件已保存: {filename}")

def analyze_pointcloud_quality(points, colors=None):
    """
    分析点云质量
    """
    print(f"\n=== 点云质量分析 ===")
    print(f"总点数: {len(points):,}")
    print(f"点云边界:")
    print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    # 计算点云密度
    volume = (points[:, 0].max() - points[:, 0].min()) * \
             (points[:, 1].max() - points[:, 1].min()) * \
             (points[:, 2].max() - points[:, 2].min())
    
    if volume > 0:
        density = len(points) / volume
        print(f"点云密度: {density:.2f} 点/立方单位")
    
    # 计算最近邻距离分布
    from scipy.spatial.distance import cdist
    if len(points) > 1000:  # 对于大点云，随机采样
        sample_indices = np.random.choice(len(points), 1000, replace=False)
        sample_points = points[sample_indices]
    else:
        sample_points = points
    
    distances = cdist(sample_points, sample_points)
    distances[distances == 0] = np.inf  # 排除自身距离
    nearest_distances = np.min(distances, axis=1)
    
    print(f"最近邻距离统计:")
    print(f"  平均: {np.mean(nearest_distances):.6f}")
    print(f"  标准差: {np.std(nearest_distances):.6f}")
    print(f"  最小: {np.min(nearest_distances):.6f}")
    print(f"  最大: {np.max(nearest_distances):.6f}")

if __name__ == "__main__":
    img_path = "E:/EQA/unrealzoo_gym/example/depth_image/openai_gpt4o_2024_11_20/img"
    save_dir = "E:/EQA/unrealzoo_gym/example/point_cloud"
    os.makedirs(save_dir, exist_ok=True)
    env_list = [
        "SuburbNeighborhood_Day",
    #     "ModularNeighborhood",
        # "ModularSciFiVillage", 
        # "Cabin_Lake",
        # "Pyramid",
        # "RuralAustralia_Example_01",
        # "ModularVictorianCity",
        # "Map_ChemicalPlant_1"
    ]
    for env_name in env_list:
        depth_img = np.load(os.path.join(img_path, f"{env_name}_depth.npy"))  # (H, W) 深度图
        rgb_img = cv2.imread(os.path.join(img_path, f"{env_name}_rgb.png"))  # (H, W, 3) RGB图像
        camera_intrinsics = {
                'fx': 256,  # 近似焦距
                'fy': 256,
                'cx': 256,
                'cy': 256
            }
        # points, colors = depth_to_pointcloud(
        #     depth_image=depth_img,
        #     camera_intrinsics=camera_intrinsics,
        #     rgb_image=rgb_img,
        #     depth_scale=1.0  # 根据您的深度单位调整
        # )

        points, colors = reconstruct_pointcloud_from_observation(rgb_img, depth_img, save_dir=save_dir, env_name=env_name)
        visualize_pointcloud_open3d(points, colors)