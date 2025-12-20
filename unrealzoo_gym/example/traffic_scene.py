import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
import argparse
#import gym_unrealcv
import gymnasium as gym
# from gymnasium import wrappers
import cv2
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, agents, augmentation, configUE, sample_agent_configs
import json
from pynput import keyboard
from datetime import datetime
import time
import numpy as np
import json
import os
import random
from dotenv import load_dotenv
from openai import OpenAI
import transforms3d
import math
import threading

load_dotenv(override=True)

unreal_lock = threading.Lock()

def update_json_file(file_path, new_entries):
    """
    Reads a JSON file, updates its content with new entries, and then writes it back.
    If the file does not exist or contains invalid JSON, a new file will be created or overwritten.

    Args:
        file_path (str): The path to the JSON file.
        new_entries (dict): A dictionary of entries to add or update in the JSON file.
    """
    existing_data = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                if content.strip(): # Ensure the file is not empty
                    existing_data = json.loads(content)
        except json.JSONDecodeError:
            print(f"Warning: File '{file_path}' is not a valid JSON file or is corrupted. New content will be created.")
        except Exception as e:
            print(f"Error reading file '{file_path}': {e}. Will attempt to create new content.")
    
    existing_data.update(new_entries)
    
    try:
        with open(file_path, 'w') as f:
            json.dump(existing_data, f, indent=4)
        # print(f"Data successfully updated in '{file_path}'.") # Can be uncommented if needed
    except Exception as e:
        print(f"Error writing to file '{file_path}': {e}")

from example.agent_configs_sampler import AgentSampler, GraphBasedSampler

def check_reach(goal_loca,cur_loac):
    distance = np.linalg.norm(np.array(goal_loca[:2]) - np.array(cur_loac[:2]))
    return distance > 200

def caculate_relative_pose( cur_obj_pose,tar_obj_pose):
        direction_vector = np.array(tar_obj_pose[:3]) - np.array(cur_obj_pose[:3])
        pitch,yaw,roll = cur_obj_pose[3:]
        pitch = np.radians(pitch)
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
        roll = np.radians(roll)
        rot = transforms3d.euler.euler2mat(-roll, -pitch, yaw, 'sxyz')
        rot_wl = np.linalg.inv(rot)
        # Transform the direction vector to obj1's frame
        local_direction = np.dot(rot_wl, direction_vector)
        return [local_direction[1],local_direction[0]]

def calculate_distance(pos1, pos2):
    """计算两个位置之间的欧几里得距离"""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)


def save_data(data, img_list, trace_dir):
    import json
    import os
    # create base directory with timestamp if not provided in trace_dir
    # Assuming trace_dir is already a specific path for this agent/action
    if not os.path.exists(trace_dir):
        os.makedirs(trace_dir)
    
    # save data to json file
    with open(os.path.join(trace_dir, 'data.json'), 'w') as f:
        json.dump(data, f, indent=4)
    
    # save images
    obs_dir = os.path.join(trace_dir, 'obs')
    if not os.path.exists(obs_dir):
        os.makedirs(obs_dir)
    
    for i, img in enumerate(img_list):
        cv2.imwrite(os.path.join(obs_dir, f'image_{i}.png'), img)
    print(f"Data saved to {trace_dir}")

def save_environment_snapshot(unwrapped_env, lock):
    """保存当前场景中所有 Agent 的位置和旋转"""
    snapshot = {}
    with lock:
        # 获取所有相关对象列表 (player, car, etc.)
        # 这里假设 unwrapped_env.agents 包含了所有感兴趣的对象
        all_agents = list(unwrapped_env.agents.keys())
        
        for agent in all_agents:
            try:
                loc = unwrapped_env.unrealcv.get_obj_location(agent)
                rot = unwrapped_env.unrealcv.get_obj_rotation(agent)
                snapshot[agent] = {
                    "location": loc,
                    "rotation": rot
                }
            except Exception as e:
                print(f"Warning: Could not snapshot agent {agent}: {e}")
    return snapshot

def run_animation_sequence_in_background(unwrapped_env, agent_name, target_map, animation_sequence, 
                                       stop_event, snapshot_event, resume_event, lock):
    """
    执行动作序列。
    target_map: {step_index: target_name} 的字典，指定某一步动作对应的交互对象。
    """
    print(f"[Thread-Animation] {agent_name} 启动动作序列: {animation_sequence}")
    
    if isinstance(animation_sequence, str):
        animation_sequence = [animation_sequence]
        
    total_steps = len(animation_sequence)
    pick_up_flag = False
    for step_idx, animation_type in enumerate(animation_sequence):
        if stop_event.is_set(): break

        # --- 获取当前步骤对应的目标 ---
        target_name = target_map.get(step_idx) # 如果这一步不需要目标，则为 None

        print(f"[Thread-Animation] {agent_name} 执行动作 {step_idx+1}/{total_steps}: {animation_type} (Target: {target_name})")
        
        # --- 1. 执行动作逻辑 (加锁) ---
        pick_up_class = "BP_GrabMoveDrop_C" 

        if animation_type == 'pick_up':
            with lock:
                if not pick_up_flag:
                    loca = unwrapped_env.unrealcv.get_obj_location(agent_name)
                    rot = unwrapped_env.unrealcv.get_obj_rotation(agent_name)
                    theta = np.deg2rad(rot[1])
                    bias = [50*np.cos(theta-np.pi/2), 50*np.sin(theta-np.pi/2), 0]
                    loc = [loca[i] + bias[i] for i in range(3)]
                    unwrapped_env.unrealcv.new_obj(pick_up_class, target_name, loc, rot)
                    unwrapped_env.unrealcv.set_obj_color(target_name, np.random.randint(0, 255, 3))
                    pick_up_flag = True

            time.sleep(1)
            with lock:
                unwrapped_env.unrealcv.set_animation(agent_name, animation_type)
            time.sleep(2.0)

        elif animation_type == 'in_vehicle':
            if target_name:
                vehicle = target_name
                with lock:
                    unwrapped_env.unrealcv.set_max_speed(agent_name, 100)
                    loca = unwrapped_env.unrealcv.get_obj_location(vehicle)
                    rot = unwrapped_env.unrealcv.get_obj_rotation(vehicle)
                    theta = np.deg2rad(rot[1])
                    bias = [200*np.cos(theta+np.pi/2), 200*np.sin(theta+np.pi/2), 0]
                    loc = [loca[i] + bias[i] for i in range(3)]
                    # unwrapped_env.unrealcv.set_obj_location(agent_name, loc)

                    unwrapped_env.unrealcv.nav_to_goal_bypath(agent_name, loc)
                time.sleep(0.5)
                with lock:
                    unwrapped_env.unrealcv.set_obj_rotation(agent_name, rot)
                time.sleep(0.5)
                with lock:
                    unwrapped_env.unrealcv.set_animation(agent_name, 'enter_vehicle')
            time.sleep(3.0)

        else:
            # 普通动画
            with lock:
                unwrapped_env.unrealcv.set_animation(agent_name, animation_type)
            time.sleep(3.0)

        # --- 2. 关键点：第一个动作结束后的同步 ---
        if step_idx == 0:
            print(f"[Thread-Animation] {agent_name} 第一个动作完成，通知主线程保存快照...")
            snapshot_event.set() # 1. 通知主线程：我已经做完第一个动作了
            
            print(f"[Thread-Animation] {agent_name} 等待主线程快照完成...")
            resume_event.wait()  # 2. 阻塞等待：等主线程存完数据叫我继续
            print(f"[Thread-Animation] {agent_name} 恢复执行后续动作...")
        
        # 动作间隔
        time.sleep(0.5)

    print(f"[Thread-Animation] {agent_name} 动作序列结束。")

def calculate_look_at_rotation(source_loc, target_loc):
    """
    计算从 source_loc 看向 target_loc 所需的 UnrealCV 旋转 (Pitch, Yaw, Roll)
    """
    dx = target_loc[0] - source_loc[0]
    dy = target_loc[1] - source_loc[1]
    dz = target_loc[2] - source_loc[2]
    
    distance_xy = math.sqrt(dx*dx + dy*dy)
    
    # 计算 Yaw (水平旋转)
    yaw = math.degrees(math.atan2(dy, dx))
    
    # 计算 Pitch (垂直旋转)
    # 注意：在 Unreal 中，看向下方通常是负 Pitch，但在某些坐标系可能是正。
    # atan2(dz, dist) 算出的是仰角，看向下方物体 dz 为负，结果为负。
    pitch = math.degrees(math.atan2(dz, distance_xy))
    
    return [pitch, yaw, 0.0]

def get_optimal_camera_pose(unwrapped_env, agent_name, target_map, lock):
    """
    根据 Agent 和目标物体的位置，计算最佳第三视角相机位置和旋转。
    """
    points = []
    
    # 1. 获取 Agent 起始位置
    with lock:
        agent_loc = unwrapped_env.unrealcv.get_obj_location(agent_name)
    points.append(np.array(agent_loc))
    
    # 2. 获取所有交互目标的位置
    for step_idx, target_name in target_map.items():
        if target_name:
            with lock:
                try:
                    t_loc = unwrapped_env.unrealcv.get_obj_location(target_name)
                    points.append(np.array(t_loc))
                except:
                    print(f"Warning: Could not get location for target {target_name}")

    if not points:
        return None, None

    # 3. 计算几何中心 (Camera LookAt Point)
    points_np = np.array(points)
    center = np.mean(points_np, axis=0)
    
    # 4. 计算场景跨度 (用于决定相机距离)
    # 计算所有点到中心的距离，取最大值
    distances = np.linalg.norm(points_np - center, axis=1)
    max_dist = np.max(distances)
    
    # 基础距离参数
    min_camera_dist = 250.0 # 最小距离 (cm)
    scale_factor = 2.0      # 距离缩放因子
    camera_dist = max(min_camera_dist, max_dist * scale_factor)
    
    # 5. 确定相机方位
    # 策略：如果有点的移动，取垂直于移动向量的方向；如果是原地动作，取默认前方偏移。
    if len(points) > 1:
        # 动作向量：起点 -> 终点
        start_pt = points[0]
        end_pt = points[-1]
        move_vec = end_pt - start_pt
        
        # 如果移动距离太小，视为原地
        if np.linalg.norm(move_vec) < 50:
            offset_dir = np.array([1, 1, 0]) # 默认 45度
        else:
            # 计算垂直向量 (-y, x) 得到侧面视角
            offset_dir = np.array([-move_vec[1], move_vec[0], 0])
    else:
        # 只有一个点（原地动作），默认放在侧前方
        offset_dir = np.array([1, 1, 0])

    # 归一化方向向量
    norm = np.linalg.norm(offset_dir)
    if norm > 0:
        offset_dir = offset_dir / norm
    else:
        offset_dir = np.array([1, 0, 0])

    # 6. 计算最终相机位置
    # 位置 = 中心 + 方向偏移 * 距离 + 高度偏移
    camera_height = max(800, camera_dist * 0.5) # 距离越远，相机越高
    
    cam_x = center[0] + offset_dir[0] * camera_dist
    cam_y = center[1] + offset_dir[1] * camera_dist
    cam_z = center[2] + camera_height
    
    final_cam_loc = [cam_x, cam_y, cam_z]
    
    # 7. 计算旋转 (Look At Center)
    final_cam_rot = calculate_look_at_rotation(final_cam_loc, center)
    
    print(f"[Camera-Algo] Center: {center}, Radius: {max_dist}, CamDist: {camera_dist}")
    return final_cam_loc, final_cam_rot




if __name__ == '__main__':
    # env name
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=0, help='random seed')
    parser.add_argument("-t", '--time-dilation', dest='time_dilation', default=-1, help='time_dilation to keep fps in simulator')
    parser.add_argument("-n", '--nav-agent', dest='nav_agent', action='store_true', help='use nav agent to control the agents')
    parser.add_argument("-d", '--early-done', dest='early_done', default=-1, help='early_done when lost in n steps')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')
    parser.add_argument("-c", '--ifonly-counting', dest='if_cnt', action='store_true', help='if only counting the number of agents in the scene')
    parser.add_argument("--config-path", dest='config_path', default=os.path.join(current_dir, "solution"), help='path to model config file')
    parser.add_argument("--model", dest="model", default="gemini_pro", help="model name")
    parser.add_argument("--camera_height", dest="camera_height", type=int, default=1200, help="camera height from the ground")
    parser.add_argument("-p", "--pid_port", type=int, default=50007, help="UnrealCV watchdog pid")
    parser.add_argument("--use_pid", type=bool,default=False, help="Whether to use pid watchdog to monitor the UE process")
    parser.add_argument("--ue_log_dir", default=os.path.join(current_dir, "unreal_log_path"), help="unreal engine logging directory")
    parser.add_argument('--status_path', default=f"{current_dir}/GT_test", help="场景路径")
    parser.add_argument("--use_adaptive", type=bool, default=True, help="是否使用自适应采样器")
    args = parser.parse_args()

    env_dict = {
        "SuburbNeighborhood_Day":{
            "height": 0,
            "agent_categories": ['player','drone','car', "robotdog"]
        },
        # "FlexibleRoom":{
        #     "height": 0,
        #     "agent_categories": ['player','drone','car', "robotdog"]
        # }
    }

    all_type_ranges = {
        'player': (5, 5),
        'car': (3, 3),
        'drone': (0, 0),
        'animal': (0, 0),
        "robotdog": (0, 0),
        'motorbike': (0, 0)
    }

    for env_name in env_dict.keys():
        status_file = os.path.join(args.status_path,env_name, "status_recorder.json")
        if not os.path.exists(status_file):
            os.makedirs(os.path.join(args.status_path,env_name), exist_ok=True)
            status = {}
            with open(status_file, 'w') as f:
                json.dump(status, f)
        with open(status_file, 'r') as f:
            status = json.load(f)
        false_cnt = sum(1 for v in status.values() if v is False)
        if false_cnt < 10:
            agent_categories = env_dict[env_name]["agent_categories"]
            type_ranges = {k: v for k, v in all_type_ranges.items() if k in agent_categories}
            min_total = sum(v[0] for v in type_ranges.values())
            max_total = sum(v[1] for v in type_ranges.values())
            env_id = f'UnrealCvEQA_DATA-{env_name}-DiscreteRgbd-v0'
            env = gym.make(env_id)
            obj_2_hide = []
            if env_name == "SuburbNeighborhood_Day":
                obj_2_hide = ["BP_Tree_Skinned_LargeSplit2", "BP_Tree_Skinned_LargeSplit3", "BP_Tree_Skinned_LargeSplit4", "BP_Tree_Skinned_LargeSplit5", "BP_Tree_Skinned_LargeSplit6",
                            "BP_Tree_Skinned_LargeSplit7", "BP_Tree_Skinned_LargeSplit_6", "BP_Tree_Skinned_Large2_2", "BP_Tree_Skinned_Large3", "BP_Tree_Skinned_Large4",
                            "BP_Tree_Skinned_Large5", "BP_Tree_Skinned_Large6", "BP_Tree_Skinned_Large_9"]
            elif env_name == "Venice":
                obj_2_hide = [
                    "BP_tree10_49","BP_tree11_52","BP_tree12_5","BP_tree13_8","BP_tree14_27","BP_tree15_19",
                    "BP_tree16_30","BP_tree17","BP_tree18_2","BP_tree19_5","BP_tree20_8","BP_tree21_11",
                    "BP_tree22_14","BP_tree23_17","BP_tree24_20","BP_tree25_23","BP_tree26_32",
                    "BP_tree27_29","BP_tree28_35","BP_tree29_38","BP_tree2_5","BP_tree30_41",
                    "BP_tree31_47","BP_tree32_50","BP_tree33_53","BP_tree34_59","BP_tree35_62",
                    "BP_tree37_74","BP_tree38_2","BP_tree39_5","BP_tree3_14","BP_tree40_8",
                    "BP_tree41_11","BP_tree42_14","BP_tree43_17","BP_tree44_20","BP_tree45_32",
                    "BP_tree46_35","BP_tree47_38","BP_tree48_41","BP_tree49_44","BP_tree4_25",
                    "BP_tree50_47","BP_tree51_50","BP_tree54_29","BP_tree5_28","BP_tree6_31",
                    "BP_tree7_37","BP_tree8_40","BP_tree9_46","BP_tree_20","BP_tree36_71"
                ]
            # some configs
            graph_path = f"./agent_configs_sampler/points_graph/{env_name}/environment_graph_1.gpickle"
            graph_path = os.path.join(current_dir, graph_path)
            # use for event plot
            bodyshot_path = os.path.join(current_dir, "./agent_caption/agent_render")
            name_dict_path = os.path.join(current_dir, "./agent_configs_sampler/agent_caption/agent_name.json")
            
            batch = datetime.now().strftime("%m%d-%H%M%S")
            obs_name = "BP_Character_C_1"
            action = [-1]
            # env wrappers
            if int(args.time_dilation) > 0:  # -1 means no time_dilation
                env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
            if int(args.early_done) > 0:  # -1 means no early_done
                env = early_done.EarlyDoneWrapper(env, int(args.early_done))
            if args.monitor:
                env = monitor.DisplayWrapper(env)
            
            env = sample_agent_configs.SampleAgentConfigWrapper(
                env,
                agent_category=agent_categories,
                camera_height= args.camera_height,
                model=args.model,   
                min_types=1,  
                max_types=len(agent_categories),
                type_count_ranges=type_ranges,       
                min_total_agents=min_total,  
                max_total_agents=max_total,
                graph_path=graph_path,
                if_cnt=args.if_cnt,
                config_path=args.config_path,
                obj_2_hide=obj_2_hide,
                bodyshot_path=bodyshot_path,
                name_dict_path=name_dict_path,
                use_adaptive=args.use_adaptive,
                normal_variance_threshold=0.1,      # 从 0.05 -> 0.1 (更宽松的光滑度)
                slope_threshold=0.5,                # 从 0.866 -> 0.5 (60°，更宽松的坡度)
                safety_margin_cm=30                 # 从 50 -> 30 (更紧凑的安全边距)
            )

            env = configUE.ConfigUEWrapper(env, offscreen=False,resolution=(1000,1000))
            if args.ue_log_dir:
                os.makedirs(args.ue_log_dir, exist_ok=True)
                env.ue_log_path = args.ue_log_dir
            # pid config
            if args.use_pid:
                env.unwrapped.send_pid = True
                env.unwrapped.watchdog_port = args.pid_port
            
            save_cnt = 0
            try:
                while save_cnt <= 21:
                    state, info = env.reset()
                    agent_num = len(list(env.unwrapped.agents.keys()))
                    actions = action*(agent_num + 1)
                    print("state shape:", state.shape)
                    obj_dict = info['object_dict']
                    if obj_dict == {}:
                        print("no agent in the scene")
                        continue
                    current_target_configs = env.unwrapped.target_configs
                    safe_start = env.unwrapped.safe_start[0]
                    
                    instance_id = info["batch_id"]
                    if args.if_cnt is False:
                        base_gt_path = os.path.join(current_dir, f"GT_test/{env_name}")
                    else:
                        base_gt_path = os.path.join(current_dir, f"GT_test/{env_name}_counting_only")
                    current_gt_info = os.path.join(base_gt_path, str(instance_id)) # including gt_info.json and obs
                    current_instance_obs_path = os.path.join(current_gt_info , f"obs")
                    gt_info_filename = os.path.join(current_gt_info, f"gt_info.json")
                    record_file = os.path.join(base_gt_path, f"status_recorder.json")
                    trace_dir = os.path.join(current_gt_info, f"trace")
                    player_info = current_target_configs["player"]
                    unwrapped_env = env.unwrapped
                    # update_camera_configs
                    env.unwrapped.unrealcv.cam = env.unwrapped.unrealcv.get_camera_config()
                    env.unwrapped.update_camera_assignments()
                    # cam_id = env.unwrapped.agents[obs_name]['cam_id']
                    # sample obs
                    collected_images_for_instance = []  # record obs
                    cam_position = env.unwrapped.camera_position
                    print(f"\nProcessing Batch: {batch}, Instance ID: {instance_id}")
                    trace_cnt = 0
                    safe_start_to_collect = []
                    cur_pose = cam_position[0]
                    cur_location = cur_pose[0:3]
                    cur_rotation = cur_pose[3: ]
                    env.unwrapped.unrealcv.cam = env.unwrapped.unrealcv.get_camera_config()
                    env.unwrapped.update_camera_assignments()
                    tp_cam_id = env.unwrapped.vacant_cam_id[1] 
                    side_cam_id = env.unwrapped.vacant_cam_id[2]
                    print(f"[Main-Loop] 俯视相机 ID: {tp_cam_id}, 侧视相机 ID: {side_cam_id}")
                    recording_info = info['recording_info']

                    # --- 1. 同步相机位姿 (确保录制连续性) ---
                    if recording_info:
                        tp_pose = recording_info.get('tp_cam_pose')
                        side_pose = recording_info.get('side_cam_pose')
                        
                        with unreal_lock:
                            if tp_pose:
                                env.unwrapped.unrealcv.set_cam_location(tp_cam_id, tp_pose[:3])
                                env.unwrapped.unrealcv.set_cam_rotation(tp_cam_id, tp_pose[3:])
                            if side_pose:
                                env.unwrapped.unrealcv.set_cam_location(side_cam_id, side_pose[:3])
                                env.unwrapped.unrealcv.set_cam_rotation(side_cam_id, side_pose[3:])
                        print(f"[Main-Loop] 已同步相机位姿 (Start Frame: {recording_info.get('last_frame_idx')})")

                    # --- 2. 筛选离相机最近的人和车 ---
                    # 获取当前俯视相机位置作为参考点
                    cam_loc = env.unwrapped.unrealcv.get_cam_location(side_cam_id)
                    
                    all_agents = env.unwrapped.target_list
                    # 简单的名称过滤
                    players = [a for a in all_agents if 'player' in a]
                    cars = [a for a in all_agents if 'car' in a]
                    
                    def get_closest_agent(agent_list, ref_loc):
                        if not agent_list: return None
                        closest = None
                        min_dist = float('inf')
                        for agent in agent_list:
                            # 获取对象位置
                            loc = env.unwrapped.unrealcv.get_obj_location(agent)
                            # calculate_distance 已经在文件中定义
                            dist = calculate_distance(loc, ref_loc)
                            if dist < min_dist:
                                min_dist = dist
                                closest = agent
                        return closest

                    selected_agents = []
                    p = get_closest_agent(players, cam_loc)
                    c = get_closest_agent(cars, cam_loc)
                    
                    if p: selected_agents.append(p)
                    if c: selected_agents.append(c)

                    if not selected_agents:
                        print("[Main-Loop] 附近没有人或车，跳过导航演示。")
                    else:
                        print(f"[Main-Loop] 选中最近的导航对象: {selected_agents}")
                        # --- 恢复录制信息 ---
                        frame_idx = recording_info.get('last_frame_idx', 0)
                        tp_dir = recording_info.get('tp_frames_dir')
                        side_dir = recording_info.get('side_frames_dir')
                        # --- 3. 设定目标并导航 ---
                        navigation_tasks = []
                        for agent_name in selected_agents:
                            curr_loc = env.unwrapped.unrealcv.get_obj_location(agent_name)
                            env.unwrapped.unrealcv.set_max_speed(agent_name, 50)
                            # env.unwrapped.unrealcv.set_phy(agent_name, 1)
                            # 判断是否是车 (根据名字包含 car 或 vehicle)
                            if 'car' in agent_name.lower():
                                # 获取当前旋转，计算前方向量
                                curr_rot = env.unwrapped.unrealcv.get_obj_rotation(agent_name)
                                yaw = curr_rot[1] # Unreal Rotation: [Pitch, Yaw, Roll]
                                yaw_rad = math.radians(yaw)
                                move_dist = 1500.0 
                                target_x = curr_loc[0] + move_dist * math.cos(yaw_rad)
                                target_y = curr_loc[1] + move_dist * math.sin(yaw_rad)
                                target_loc = [target_x, target_y, curr_loc[2]]
                                print(f"[Main-Loop] 车辆 {agent_name} 沿朝向(Yaw={yaw:.1f}) 移动至 {target_loc}")
                                
                            else:
                                # 人：保持随机移动逻辑
                                rand_x = curr_loc[0] + random.uniform(-1000, 1000)
                                rand_y = curr_loc[1] + random.uniform(-1000, 1000)
                                target_loc = [rand_x, rand_y, curr_loc[2]]
                                print(f"[Main-Loop] 行人 {agent_name} 随机移动至 {target_loc}")
                                
                            navigation_tasks.append((agent_name, target_loc))
                        for agent_name, t_loc in navigation_tasks:
                            if 'car' in agent_name.lower():
                                env.unwrapped.unrealcv.set_phy(agent_name, 1)
                            env.unwrapped.unrealcv.nav_to_goal(agent_name, t_loc)
                        
                        for _ in range(500):
                            tp_img = env.unwrapped.unrealcv.read_image(tp_cam_id, 'lit')
                            side_img = env.unwrapped.unrealcv.read_image(side_cam_id, 'lit')
                            tp_fname = os.path.join(tp_dir, f"frame_{frame_idx:05d}.png")
                            side_fname = os.path.join(side_dir, f"frame_{frame_idx:05d}.png")
                            cv2.imwrite(tp_fname, tp_img)            
                            cv2.imwrite(side_fname, side_img)
                            frame_idx += 1
                        print(f"[Main-Loop] 导航录制结束。")
                                      
                env.close()
                exit(-1)
            except KeyboardInterrupt:
                print('\nExiting due to KeyboardInterrupt...')
                env.close()
            except Exception as e:
                import traceback
                traceback.print_exc()
            finally:
                env.close()
                exit(-1)




