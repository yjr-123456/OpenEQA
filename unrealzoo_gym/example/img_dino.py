import argparse
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)  # Add parent directory to sys.path
import gymnasium as gym
from gymnasium import wrappers
import cv2
import time
import numpy as np
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, agents, augmentation, configUE
from example.agent_configs_sampler.agent_sample_agent import AgentSampler
import json
import random
import math
from openai import OpenAI
from dotenv import load_dotenv
import base64
import io
from PIL import Image
load_dotenv(override=True)
import transforms3d
# print("Current Directory:", os.getcwd())

def obs_transform(obs, agent_id = 0):
    obs_rgb = cv2.cvtColor(obs[agent_id][..., :3], cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    obs_depth = obs[agent_id][..., 3:]
    return obs_rgb, obs_depth

def get_camera_matrix(W, H, fov_deg):
    fov_rad = np.deg2rad(fov_deg)
    f = (W / 2) / np.tan(fov_rad / 2)
    K = np.array([[f, 0, W / 2],
                  [0, f, H / 2],
                  [0, 0, 1]])
    return K

def pose_to_matrix(cam_pose):
    tx, ty, tz, rx, ry, rz = cam_pose
    pitch = np.radians(rx)
    yaw = np.radians(ry)
    roll = np.radians(rz)
    
    # 使用正确的 Unreal 到 OpenCV 坐标系转换
    R_unreal = transforms3d.euler.euler2mat(-roll, -pitch, yaw, 'sxyz')
    # Unreal (X前,Y右,Z上) -> OpenCV (X右,Y下,Z前)
    # coord_transform = np.array([
    #     [0, 1, 0],   # X_cv = Y_unreal (右)
    #     [0, 0, -1],  # Y_cv = -Z_unreal (下)
    #     [1, 0, 0]    # Z_cv = X_unreal (前)
    # ])
    # R_mat = coord_transform @ R_unreal
    
    tvec = np.array([tx, ty, tz], dtype=np.float32)
    return R_unreal, tvec

def world_2_image(point_world, cam_pose, K):
    R_mat, tvec = pose_to_matrix(cam_pose)
    # 如果R_mat是世界到相机的旋转矩阵，直接使用
    point_cam = R_mat @ (point_world - tvec)  # 改为直接使用R_mat
    pixel = K @ point_cam.reshape(3, 1)
    u = pixel[0, 0] / pixel[2, 0]
    v = pixel[1, 0] / pixel[2, 0]
    return int(u), int(v)

def image_2_world(u, v, depth, cam_pose, K):
    R_mat, tvec = pose_to_matrix(cam_pose)
    K_inv = np.linalg.inv(K)
    pixel_homo = np.array([u, v, 1.0])
    point_cam_normalized = K_inv @ pixel_homo
    point_cam = point_cam_normalized * depth
    # 如果R_mat是世界到相机的旋转矩阵
    point_world = R_mat.T @ point_cam + tvec  # 改为使用R_mat.T
    return point_world


def calculate_depth(point_world, cam_pose):
    """
    Calculate depth of a world point from camera
    """
    R_mat, tvec = pose_to_matrix(cam_pose)
    # 保持与 world_2_image 函数一致的变换方式
    point_cam = R_mat @ (point_world - tvec)  # 使用相同的变换
    depth = point_cam[2]
    return depth

def project_points_to_image(world_points, cam_pose, W, H, fov_deg=90):
    """
    Project multiple world points to image coordinates
    
    Args:
        world_points: (N, 3) array of world coordinates
        cam_pose: camera pose [tx, ty, tz, rx, ry, rz]
        W, H: image width and height
        fov_deg: field of view in degrees
    
    Returns:
        image_points: (N, 2) array of pixel coordinates
        valid_mask: (N,) boolean array indicating which points are within image bounds
        depths: (N,) array of depth values
    """
    K = get_camera_matrix(W, H, fov_deg)
    
    image_points = []
    depths = []
    valid_mask = []
    
    for point_world in world_points:
        # 计算深度
        depth = calculate_depth(point_world, cam_pose)
        depths.append(depth)
        
        # 投影到图像
        if depth > 0:  # 只投影在相机前方的点
            u, v = world_2_image(point_world, cam_pose, K)
            
            # 检查是否在图像边界内
            if 0 <= u < W and 0 <= v < H:
                valid_mask.append(True)
            else:
                valid_mask.append(False)
                
            image_points.append([u, v])
        else:
            # 深度为负，点在相机后方
            valid_mask.append(False)
            image_points.append([-1, -1])  # 无效坐标
    
    return np.array(image_points), np.array(valid_mask), np.array(depths)

def visualize_projected_points(obs_rgb, world_points, cam_pose, W, H, fov_deg=90):
    """
    在图像上可视化投影的点
    """
    image_points, valid_mask, depths = project_points_to_image(world_points, cam_pose, W, H, fov_deg)
    
    result_img = obs_rgb.copy()
    
    for i, (point_img, is_valid, depth) in enumerate(zip(image_points, valid_mask, depths)):
        u, v = point_img
        
        if is_valid:
            # 在图像内的点用绿色标记
            cv2.circle(result_img, (int(u), int(v)), 5, (0, 255, 0), -1)
            cv2.putText(result_img, f"{i}", (int(u)+5, int(v)-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            if depth > 0:
                # 在图像外但在相机前方的点，显示在边界
                u_clamped = max(0, min(W-1, int(u)))
                v_clamped = max(0, min(H-1, int(v)))
                cv2.circle(result_img, (u_clamped, v_clamped), 3, (0, 0, 255), -1)  # 红色
                cv2.putText(result_img, f"{i}", (u_clamped+5, v_clamped-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            # 在相机后方的点不显示
    
    return result_img, image_points, valid_mask, depths

def debug_projection(world_points, cam_pose, W=1080, H=1080):
    """调试投影过程"""
    print("=== Debug Projection ===")
    print(f"Camera pose: {cam_pose}")
    
    # 检查相机朝向
    R_mat, tvec = pose_to_matrix(cam_pose)
    camera_forward = R_mat[:, 2]
    print(f"Camera forward direction: {camera_forward}")
    print(f"Expected for downward: [0, 0, -1] or close to it")
    
    # 测试第一个点（地面中心点）
    test_point = world_points[0]  # [-1090, -186, 0]
    print(f"\nTesting point: {test_point}")
    
    # 计算预期深度（几何计算）
    expected_depth = cam_pose[2] - test_point[2]  # 高度差
    print(f"Expected depth (height difference): {expected_depth}")
    
    # 计算实际深度
    actual_depth = calculate_depth(test_point, cam_pose)
    print(f"Calculated depth: {actual_depth}")
    
    # 投影到图像
    u, v = world_2_image(test_point, cam_pose, get_camera_matrix(W, H, 90))
    print(f"Projected to pixel: ({u}, {v})")
    print(f"Image center: ({W//2}, {H//2})")
    
    # 检查不同FOV的影响
    for fov in [60, 90, 120]:
        K = get_camera_matrix(W, H, fov)
        u_fov, v_fov = world_2_image(test_point, cam_pose, K)
        print(f"FOV {fov}°: pixel ({u_fov}, {v_fov})")

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

def project_points_to_image_unreal(world_points, cam_pose, W, H, fov_deg=90):
    """在UnrealCV坐标系下投影多个点"""
    K = get_camera_matrix_unreal(W, H, fov_deg)
    
    image_points = []
    depths = []
    valid_mask = []
    
    for point_world in world_points:
        u, v, depth = world_2_image_unreal(point_world, cam_pose, K)
        depths.append(depth)
        
        if depth > 0 and 0 <= u < W and 0 <= v < H:
            valid_mask.append(True)
        else:
            valid_mask.append(False)
            
        image_points.append([u, v])
    
    return np.array(image_points), np.array(valid_mask), np.array(depths)

def debug_projection_unreal(world_points, cam_pose, W=1080, H=1080):
    """在UnrealCV坐标系下调试投影"""
    print("=== Debug Projection (UnrealCV coordinate system) ===")
    print(f"Camera pose: {cam_pose}")
    
    # 检查相机朝向
    R_mat, tvec = pose_to_matrix_unreal(cam_pose)
    camera_forward = R_mat[:, 0]  # UnrealCV中X轴向前
    camera_right = R_mat[:, 1]    # Y轴向右
    camera_up = R_mat[:, 2]       # Z轴向上
    
    print(f"Camera forward (X): {camera_forward}")
    print(f"Camera right (Y): {camera_right}")  
    print(f"Camera up (Z): {camera_up}")
    
    # 测试第一个点
    test_point = world_points[0]
    print(f"\nTesting point: {test_point}")
    
    # 在UnrealCV坐标系下检查
    point_cam = R_mat.T @ (test_point - tvec)
    print(f"Point in camera coordinates (UnrealCV): {point_cam}")
    print(f"Depth (X-axis): {point_cam[0]}")
    
    # 投影
    u, v, depth = world_2_image_unreal(test_point, cam_pose, get_camera_matrix_unreal(W, H, 90))
    print(f"Projected to pixel: ({u}, {v}), depth: {depth}")

def visualize_projected_points_unreal(obs_rgb, world_points, cam_pose, W, H, fov_deg=90):
    """
    在图像上可视化投影的点 (UnrealCV坐标系版本)
    """
    image_points, valid_mask, depths = project_points_to_image_unreal(world_points, cam_pose, W, H, fov_deg)
    
    result_img = obs_rgb.copy()
    
    for i, (point_img, is_valid, depth) in enumerate(zip(image_points, valid_mask, depths)):
        u, v = point_img
        
        if is_valid:
            # 在图像内的点用绿色标记
            cv2.circle(result_img, (int(u), int(v)), 5, (0, 255, 0), -1)
            cv2.putText(result_img, f"{i}", (int(u)+5, int(v)-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            if depth > 0:
                # 在图像外但在相机前方的点，显示在边界
                u_clamped = max(0, min(W-1, int(u)))
                v_clamped = max(0, min(H-1, int(v)))
                cv2.circle(result_img, (u_clamped, v_clamped), 3, (0, 0, 255), -1)  # 红色
                cv2.putText(result_img, f"{i}", (u_clamped+5, v_clamped-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            # 在相机后方的点不显示
    
    return result_img, image_points, valid_mask, depths


if __name__ == '__main__':
    env_name = 'SuburbNeighborhood_Day'
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env_id", nargs='?', default=f'UnrealCv_base-{env_name}-ContinuousRgbd-v0',
                        help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=1, help='random seed')
    parser.add_argument("-t", '--time_dilation', dest='time_dilation', default=30, help='time_dilation to keep fps in simulator')
    parser.add_argument("-d", '--early_done', dest='early_done', default=-1, help='early_done when lost in n steps')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')
    parser.add_argument("--save_path", default=os.path.join(os.path.dirname(__file__), 'Obj_info'),help="path to save object informations")
    parser.add_argument("--model", default="doubao", help="choose evaluation models")
    parser.add_argument("--config_path", default=os.path.join(current_dir, "solution"), help="configuration file path")
    parser.add_argument("--graph_path", default=os.path.join(current_dir, "agent_configs_sampler", "points_graph"), help="scene graph file path")
    args = parser.parse_args()
    print(args.env_id)
    env = gym.make(args.env_id)
    if int(args.time_dilation) > 0:  # -1 means no time_dilation
        env = time_dilation.TimeDilationWrapper(env, args.time_dilation)
    if int(args.early_done) > 0:  # -1 means no early_done
        env = early_done.EarlyDoneWrapper(env, args.early_done)
    if args.monitor:
        env = monitor.DisplayWrapper(env)
    env = configUE.ConfigUEWrapper(env, resolution=(1080,1080),offscreen=False)
    env = augmentation.RandomPopulationWrapper(env, 1, 1, random_target=False)
    # env = agents.NavAgents(env, mask_agent=True)
    # episode_count = 50
    # rewards = 0
    # done = False
    # Total_rewards = 0

    graph_path = os.path.join(args.graph_path, env_name, "environment_graph.gpickle")
    cam_id = 1
    cam_pose = [1134,-532,672,-90, 0,0]
    # world_points = np.array([
    #     [-1090, -186, 0],      # 地面中心点
    #     [-1090, -286, 0],      # 地面前方
    #     [-1090, -86, 0],       # 地面后方
    #     [-1190, -186, 0],      # 地面左侧
    #     [-990, -186, 0],       # 地面右侧
    #     [-1090, -186, 100],    # 空中100m
    #     [-1090, -186, 200],    # 空中200m
    #     [-1500, -186, 0],      # 远处地面
    #     [-500, -186, 0],       # 近处地面
    #     [-1090, -500, 0],      # 很远的前方（可能在图像外）
    # ])
    agent_sampler = AgentSampler(graph_path)
    print("======load graph nodes======")
    print(agent_sampler.node_positions)
    try:
        states, info = env.reset(seed=int(args.seed))
        env.unwrapped.unrealcv.set_cam_location(cam_id, cam_pose[:3])
        env.unwrapped.unrealcv.set_cam_rotation(cam_id, cam_pose[3:])
        time.sleep(0.5)
        obs_rgb = env.unwrapped.unrealcv.read_image(cam_id,'lit')
        image_points, valid_mask, depths = agent_sampler.project_points_to_image_unreal(
            agent_sampler.node_list, cam_pose, W=1080, H=1080)

        result_img, image_points, valid_mask, depths = agent_sampler.visualize_projected_points_unreal(
            obs_rgb, agent_sampler.node_list, cam_pose, W=1080, H=1080)        
        # 显示结果
        cv2.imshow('Projected Points', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        env.close()


    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        env.close()
    finally:
        env.close()
