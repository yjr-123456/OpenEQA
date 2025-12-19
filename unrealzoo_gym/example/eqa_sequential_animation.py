import sys
import os
import argparse
import gymnasium as gym
import cv2
import json
import time
import numpy as np
import threading
import math
from datetime import datetime

# --- 路径设置 ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
current_dir = os.path.dirname(os.path.abspath(__file__))

# --- 导入环境依赖 ---
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, configUE
from dotenv import load_dotenv

load_dotenv(override=True)

# --- 全局锁，用于保护 UnrealCV Socket 通信 ---
unreal_lock = threading.Lock()

# ==========================================
# 1. 场景配置 (Scenario Configuration)
# ==========================================
# 这里定义了场景中所有的 Agent、动作序列以及同步规则。
# 在实际应用中，这个字典可以由 LLM 生成或从 JSON 文件读取。

SCENARIO_CONFIG = {
    "agents": {
        "BP_Character_C_1": {
            "actions": ["walk to target", "wait", "walk to car", "enter_exit car"],
            "targets": {
                0: "BP_Character_C_2",  # 第一步走向 Agent 2 (模拟汇合)
                2: "BP_SUV_C_1",        # 第三步走向车
                3: "BP_SUV_C_1"         # 第四步上车
            }
        },
        "BP_Character_C_2": {
            "actions": ["walk to target", "wait", "walk to car", "enter_exit car"],
            "targets": {
                0: "BP_Character_C_1",  # 第一步走向 Agent 1
                2: "BP_SUV_C_1",
                3: "BP_SUV_C_1"
            }
        },
        "BP_Character_C_3": {
            "actions": ["walk to point_A", "crouch", "walk to point_B"],
            "targets": {
                0: "BP_Tree_Skinned_Large2_2", # 假设的一个树
                2: "BP_TrashCan_1"              # 假设的一个垃圾桶
            }
        }
    },
    # 同步规则：定义哪些 Agent 在哪一步需要等待彼此
    "synchronization": [
        {
            # 规则：Agent 1 和 Agent 2 在开始第 2 步 (walk to car) 之前必须同步
            # 意味着他们必须都完成第 0, 1 步后，才能一起去车边
            "agents": ["BP_Character_C_1", "BP_Character_C_2"],
            "step_indices": [2] 
        }
    ]
}

# ==========================================
# 2. 辅助函数
# ==========================================

def calculate_distance(pos1, pos2):
    """计算欧几里得距离"""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)

def calculate_look_at_rotation(source_loc, target_loc):
    """计算看向目标的旋转角度"""
    dx = target_loc[0] - source_loc[0]
    dy = target_loc[1] - source_loc[1]
    dz = target_loc[2] - source_loc[2]
    distance_xy = math.sqrt(dx*dx + dy*dy)
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(dz, distance_xy))
    return [pitch, yaw, 0.0]

def update_dynamic_camera(unwrapped_env, active_agents, cam_id, lock):
    """
    计算所有活跃 Agent 的中心点，并调整相机位置
    """
    points = []
    with lock:
        for agent_name in active_agents:
            try:
                loc = unwrapped_env.unrealcv.get_obj_location(agent_name)
                points.append(np.array(loc))
            except:
                pass
    
    if not points:
        return

    # 计算中心点
    points_np = np.array(points)
    center = np.mean(points_np, axis=0)
    
    # 计算最大跨度以调整缩放
    distances = np.linalg.norm(points_np - center, axis=1)
    max_dist = np.max(distances) if len(distances) > 0 else 100
    
    # 相机参数
    camera_dist = max(400.0, max_dist * 2.5) # 最小距离400
    camera_height = max(300.0, camera_dist * 0.6)
    
    # 简单的侧俯视视角 (固定偏移方向，或者根据移动方向动态调整)
    offset_dir = np.array([1, 1, 0]) 
    offset_dir = offset_dir / np.linalg.norm(offset_dir)
    
    cam_x = center[0] + offset_dir[0] * camera_dist
    cam_y = center[1] + offset_dir[1] * camera_dist
    cam_z = center[2] + camera_height
    
    final_cam_loc = [cam_x, cam_y, cam_z]
    final_cam_rot = calculate_look_at_rotation(final_cam_loc, center)
    
    with lock:
        unwrapped_env.unrealcv.set_cam_location(cam_id, final_cam_loc)
        unwrapped_env.unrealcv.set_cam_rotation(cam_id, final_cam_rot)

# ==========================================
# 3. 任务执行线程 (Worker)
# ==========================================

def run_agent_task_thread(unwrapped_env, agent_name, action_sequence, target_map, 
                          sync_barriers, lock, stop_event):
    """
    通用 Agent 任务执行器
    """
    print(f"[{agent_name}] 线程启动，计划执行 {len(action_sequence)} 个动作")
    
    for step_idx, action_raw in enumerate(action_sequence):
        if stop_event.is_set(): break
        
        # --- 1. 同步点检查 (Pre-Action) ---
        # 如果当前步骤配置了 Barrier，则在此等待其他 Agent
        if sync_barriers and step_idx in sync_barriers:
            print(f"[{agent_name}] 在步骤 {step_idx} 等待同步...")
            try:
                sync_barriers[step_idx].wait(timeout=30) # 设置超时防止死锁
                print(f"[{agent_name}] 同步完成，继续执行。")
            except threading.BrokenBarrierError:
                print(f"[{agent_name}] 同步超时或中断！")
                break

        # --- 2. 获取目标 ---
        target_obj_name = target_map.get(step_idx)
        print(f"[{agent_name}] Step {step_idx}: {action_raw} (Target: {target_obj_name})")

        # --- 3. 执行动作逻辑 ---
        try:
            if "walk to" in action_raw:
                goal_loc = None
                if target_obj_name:
                    with lock:
                        goal_loc = unwrapped_env.unrealcv.get_obj_location(target_obj_name)
                else:
                    # 如果没有指定目标，随机走一点 (示例)
                    with lock:
                        unwrapped_env.unrealcv.nav_to_random(agent_name, 500, False)
                
                if goal_loc:
                    with lock:
                        unwrapped_env.unrealcv.nav_to_goal_bypath(agent_name, goal_loc)
                    
                    # 轮询等待到达
                    start_move_time = time.time()
                    while not stop_event.is_set():
                        with lock:
                            cur = unwrapped_env.unrealcv.get_obj_location(agent_name)
                        if calculate_distance(cur, goal_loc) < 150: # 1.5米内视为到达
                            break
                        if time.time() - start_move_time > 15: # 15秒超时
                            print(f"[{agent_name}] 移动超时，强制跳过")
                            break
                        time.sleep(0.5)

            elif action_raw == "pick_up":
                # 示例：生成物体并捡起
                if target_obj_name:
                    with lock:
                        loc = unwrapped_env.unrealcv.get_obj_location(agent_name)
                        # 在前方生成物体
                        spawn_loc = [loc[0]+50, loc[1], loc[2]]
                        unwrapped_env.unrealcv.new_obj("BP_GrabMoveDrop_C", target_obj_name, spawn_loc, [0,0,0])
                time.sleep(0.5)
                with lock:
                    unwrapped_env.unrealcv.set_animation(agent_name, "pick_up")
                time.sleep(2.0)

            elif action_raw == "enter_exit car":
                with lock:
                    unwrapped_env.unrealcv.set_animation(agent_name, "enter_vehicle")
                time.sleep(3.0)
                with lock:
                    # 移出视野模拟进入
                    unwrapped_env.unrealcv.set_obj_location(agent_name, [0,0,-5000])

            elif action_raw == "crouch":
                with lock:
                    unwrapped_env.unrealcv.set_animation(agent_name, "crouch_start")
                time.sleep(1.5)
                with lock:
                    unwrapped_env.unrealcv.set_animation(agent_name, "crouch_end")
            
            elif action_raw == "wait":
                time.sleep(2.0)

            else:
                print(f"[{agent_name}] 未知动作: {action_raw}")

        except Exception as e:
            print(f"[{agent_name}] 执行动作出错: {e}")

        time.sleep(0.5) # 动作间歇

    print(f"[{agent_name}] 任务结束。")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="SuburbNeighborhood_Day")
    parser.add_argument("--pid_port", type=int, default=50007)
    parser.add_argument("--use_pid", action='store_true')
    args = parser.parse_args()

    env_id = f'UnrealCvEQA_DATA-{args.env_name}-DiscreteRgbd-v0'
    env = gym.make(env_id)
    env = configUE.ConfigUEWrapper(env, offscreen=False, resolution=(640, 480))
    
    if args.use_pid:
        env.unwrapped.send_pid = True
        env.unwrapped.watchdog_port = args.pid_port

    try:
        env.reset()
        unwrapped_env = env.unwrapped
        
        # --- 准备相机 ---
        # 1. 更新相机分配
        unwrapped_env.update_camera_assignments()
        
        # 2. 获取一个闲置的第三视角相机 (排除 ID 0)
        vacant_cams = [cid for cid in unwrapped_env.vacant_cam_id if cid != 0]
        if not vacant_cams:
            # 如果没有闲置，尝试生成一个新的
            unwrapped_env.unrealcv.set_new_camera()
            unwrapped_env.update_camera_assignments()
            vacant_cams = [cid for cid in unwrapped_env.vacant_cam_id if cid != 0]
        
        tp_cam_id = vacant_cams[0] if vacant_cams else 1
        print(f"[Main] 使用第三视角相机 ID: {tp_cam_id}")

        # --- 解析同步配置 ---
        agent_barriers = {} 
        if "synchronization" in SCENARIO_CONFIG:
            for sync_rule in SCENARIO_CONFIG["synchronization"]:
                involved_agents = sync_rule["agents"]
                step_indices = sync_rule["step_indices"]
                
                # 创建 Barrier
                barrier = threading.Barrier(len(involved_agents))
                
                for agent in involved_agents:
                    for step in step_indices:
                        if agent not in agent_barriers:
                            agent_barriers[agent] = {}
                        agent_barriers[agent][step] = barrier

        # --- 启动 Agent 线程 ---
        active_threads = []
        stop_event = threading.Event()
        active_agent_names = list(SCENARIO_CONFIG["agents"].keys())

        print(f"[Main] 开始分发任务，涉及 {len(active_agent_names)} 个智能体...")

        for agent_name, task_data in SCENARIO_CONFIG["agents"].items():
            # 检查 Agent 是否存在于环境中
            if agent_name not in unwrapped_env.agents:
                print(f"Warning: 配置中的 {agent_name} 不在当前环境中，跳过。")
                if agent_name in active_agent_names: active_agent_names.remove(agent_name)
                continue

            my_barriers = agent_barriers.get(agent_name, {})
            
            t = threading.Thread(
                target=run_agent_task_thread,
                args=(
                    unwrapped_env, 
                    agent_name, 
                    task_data["actions"], 
                    task_data["targets"], 
                    my_barriers, 
                    unreal_lock, 
                    stop_event
                )
            )
            t.start()
            active_threads.append(t)

        # --- 主循环：录制与相机控制 ---
        full_video_frames = []
        recording_start_time = time.time()
        
        print("[Main] 开始录制...")

        while any(t.is_alive() for t in active_threads):
            # 1. 动态更新相机位置 (每 5 帧更新一次，避免过于频繁抖动)
            if len(full_video_frames) % 5 == 0:
                update_dynamic_camera(unwrapped_env, active_agent_names, tp_cam_id, unreal_lock)

            # 2. 录制帧
            with unreal_lock:
                frame = unwrapped_env.unrealcv.read_image(tp_cam_id, 'lit')
            
            if frame is not None:
                # 转换颜色空间 RGB -> BGR
                full_video_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            time.sleep(0.05) # 约 20 FPS

            # 超时保护 (例如 60 秒)
            if time.time() - recording_start_time > 60:
                print("[Main] 场景执行超时，强制停止")
                break
        
        print(f"[Main] 所有任务结束，共录制 {len(full_video_frames)} 帧。")

        # --- 保存数据 ---
        save_dir = os.path.join(current_dir, "multi_agent_output")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 保存视频帧
        obs_dir = os.path.join(save_dir, "frames")
        if not os.path.exists(obs_dir):
            os.makedirs(obs_dir)
            
        for i, img in enumerate(full_video_frames):
            cv2.imwrite(os.path.join(obs_dir, f'{i:04d}.png'), img)
            
        # 保存配置元数据
        with open(os.path.join(save_dir, "scenario_config.json"), 'w') as f:
            json.dump(SCENARIO_CONFIG, f, indent=4)
            
        print(f"[Main] 数据已保存至 {save_dir}")

    except KeyboardInterrupt:
        print("[Main] 用户中断")
    except Exception as e:
        print(f"[Main] 发生错误: {e}")
    finally:
        stop_event.set()
        for t in active_threads:
            t.join()
        env.close()