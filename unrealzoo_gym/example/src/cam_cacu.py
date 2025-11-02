import numpy as np
import cv2
import transforms3d


def pose_to_matrix_unreal(cam_pose):
    """直接使用UnrealCV坐标系，不进行坐标系转换"""
    tx, ty, tz, rx, ry, rz = cam_pose
    pitch = np.radians(rx)
    yaw = np.radians(ry)
    roll = np.radians(rz)
    
    # 直接使用UnrealCV的旋转矩阵
    R_mat = transforms3d.euler.euler2mat(-roll, -pitch, yaw, 'sxyz')
    tvec = np.array([tx, ty, tz], dtype=np.float32)
    
    return R_mat, tvec

def get_camera_matrix_unreal(W, H, fov_deg):
    """适配UnrealCV坐标系的相机内参矩阵"""
    fov_rad = np.deg2rad(fov_deg)
    f = (W / 2) / np.tan(fov_rad / 2)
    
    # UnrealCV坐标系下的内参矩阵
    # 可能需要调整主点或其他参数
    K = np.array([[f, 0, W / 2],
                [0, f, H / 2],
                [0, 0, 1]])
    return K

def world_2_image_unreal(point_world, cam_pose, K):
    """在UnrealCV坐标系下进行投影"""
    R_mat, tvec = pose_to_matrix_unreal(cam_pose)

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

def calculate_depth_unreal(point_world, cam_pose):
    """在UnrealCV坐标系下计算深度"""
    R_mat, tvec = pose_to_matrix_unreal(cam_pose)
    point_cam = R_mat.T @ (point_world - tvec)
    
    # 在UnrealCV中，深度通常是X轴方向（相机前方）
    depth = point_cam[0]
    return depth

def calculate_extrinsic_unreal(cam_pose, invert=False):
    """计算UnrealCV坐标系下的相机外参矩阵
        invert=True: 世界 → 相机
        invert=False: 相机 → 世界
    """
    R_mat, tvec = pose_to_matrix_unreal(cam_pose)
    if invert:
        # 世界 → 相机（标准外参）
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R_mat.T
        extrinsic[:3, 3] = -R_mat.T @ tvec
    else:
        # 相机 → 世界（Pose）
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R_mat
        extrinsic[:3, 3] = tvec
    
    return extrinsic

def project_points_to_image_unreal(world_points, cam_pose, K, W, H):
    """在UnrealCV坐标系下投影多个点"""
    image_points = []
    depths = []
    valid_mask = []
    valid_points = []
    for point_world in world_points:
        u, v, depth = world_2_image_unreal(point_world, cam_pose, K)
        depths.append(depth)

        if depth > 0 and 5 <= u < W-5 and 5 <= v < H-5:
            valid_mask.append(True)
            valid_points.append(point_world)
        else:
            valid_mask.append(False)
            
        image_points.append([u, v])
    
    return np.array(image_points), np.array(valid_mask), np.array(depths)

