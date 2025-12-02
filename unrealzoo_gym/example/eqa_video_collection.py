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
    
    for step_idx, animation_type in enumerate(animation_sequence):
        if stop_event.is_set(): break

        # --- 获取当前步骤对应的目标 ---
        target_name = target_map.get(step_idx) # 如果这一步不需要目标，则为 None

        print(f"[Thread-Animation] {agent_name} 执行动作 {step_idx+1}/{total_steps}: {animation_type} (Target: {target_name})")
        
        # --- 1. 执行动作逻辑 (加锁) ---
        pick_up_class = "BP_GrabMoveDrop_C" 

        if animation_type == 'pick_up':
            with lock:
                loca = unwrapped_env.unrealcv.get_obj_location(agent_name)
                rot = unwrapped_env.unrealcv.get_obj_rotation(agent_name)
                theta = np.deg2rad(rot[1])
                bias = [50*np.cos(theta-np.pi/2), 50*np.sin(theta-np.pi/2), 0]
                loc = [loca[i] + bias[i] for i in range(3)]
                unwrapped_env.unrealcv.new_obj(pick_up_class, target_name, loc, rot)
                unwrapped_env.unrealcv.set_obj_color(target_name, np.random.randint(0, 255, 3))
            
            time.sleep(1)
            with lock:
                unwrapped_env.unrealcv.set_animation(agent_name, animation_type)
            time.sleep(2.0)

        elif animation_type == 'in_vehicle':
            if target_name:
                vehicle = target_name
                with lock:
                    loca = unwrapped_env.unrealcv.get_obj_location(vehicle)
                    rot = unwrapped_env.unrealcv.get_obj_rotation(vehicle)
                    theta = np.deg2rad(rot[1])
                    bias = [200*np.cos(theta+np.pi/2), 200*np.sin(theta+np.pi/2), 0]
                    loc = [loca[i] + bias[i] for i in range(3)]
                    unwrapped_env.unrealcv.set_obj_location(agent_name, loc)
                
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


if __name__ == '__main__':
    # env name
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=0, help='random seed')
    parser.add_argument("-t", '--time-dilation', dest='time_dilation', default=10, help='time_dilation to keep fps in simulator')
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
        'player': (2, 6),
        'car': (1, 1),
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
                    print("state shape:", state.shape)
                    obj_dict = info['object_dict']
                    if obj_dict == {}:
                        print("no agent in the scene")
                        continue
                    current_target_configs = env.unwrapped.target_configs
                    safe_start = env.unwrapped.safe_start[0]
                    agent_num = len(list(env.unwrapped.agents.keys()))
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
                    cam_id = env.unwrapped.agents[obs_name]['cam_id']
                    # sample obs
                    collected_images_for_instance = []  # record obs
                    cam_position = env.unwrapped.camera_position
                    print(f"\nProcessing Batch: {batch}, Instance ID: {instance_id}")
                    trace_cnt = 0
                    safe_start_to_collect = []
                    cur_pose = cam_position[0]
                    cur_location = cur_pose[0:3]
                    cur_rotation = cur_pose[3: ]

                    
                    # obj animation and recording
                    for agent_name, animation_data in zip(player_info['name'], player_info['animation']):
                        env.unwrapped.unrealcv.set_obj_location(obs_name, cur_location)
                        env.unwrapped.unrealcv.set_obj_rotation(obs_name, cur_rotation)
                        # 确保是列表
                        animation_sequence = animation_data if isinstance(animation_data, list) else [animation_data]
                        
                        print(f"\n[Main-Loop] 开始处理角色 '{agent_name}' 的意图序列: {animation_sequence}")

                        # --- 准备同步信号 ---
                        stop_event = threading.Event()
                        snapshot_event = threading.Event() # 线程 -> 主程序 
                        resume_event = threading.Event()   # 主程序 -> 线程 

                        # --- 预处理 target_map ---
                        target_map = {}
                        
                        for idx, action_type in enumerate(animation_sequence):
                            if action_type == 'pick_up':
                                pick_up_class = "BP_GrabMoveDrop_C"
                                # 加上 idx 防止同一个序列多次捡东西名字冲突
                                pick_up_name = f"{pick_up_class}_{agent_name}_{idx}" 
                                unwrapped_env.pickup_list.append(pick_up_name)
                                target_map[idx] = pick_up_name
                                
                            elif action_type == 'in_vehicle':
                                agent_type = 'car'
                                has_vehicle = any(unwrapped_env.agents[obj]['agent_type'] in ['car', 'motorbike'] for obj in unwrapped_env.target_list)
                                vehicle_list = [obj for obj in unwrapped_env.player_list if has_vehicle and unwrapped_env.agents[obj]['agent_type'] in ['car', 'motorbike']]   
                                
                                if len(vehicle_list) > 0:
                                    with unreal_lock:
                                        obj_loc = unwrapped_env.unrealcv.get_obj_location(agent_name)
                                    min_distance = float('inf')
                                    nearest_vehicle = None
                                    for vehicle_name in vehicle_list:
                                        with unreal_lock:
                                            vehicle_loc = unwrapped_env.unrealcv.get_obj_location(vehicle_name)
                                        distance = calculate_distance(obj_loc, vehicle_loc)
                                        if distance < min_distance:
                                            min_distance = distance
                                            nearest_vehicle = vehicle_name
                                    
                                    if nearest_vehicle:
                                        target_map[idx] = nearest_vehicle
                                    else:
                                        print(f"Warning: Found vehicle list but calculation failed for {agent_name} step {idx}.")
                                else:
                                    print(f"Warning: No vehicle found for {agent_name} at step {idx}. Action may fail.")

                        # --- 启动后台线程 ---
                        animation_thread = threading.Thread(
                            target=run_animation_sequence_in_background,
                            args=(unwrapped_env, agent_name, target_map, animation_sequence, 
                                  stop_event, snapshot_event, resume_event, unreal_lock)
                        )
                        animation_thread.start()

                        # --- 移动观察者到合适位置 (可选，这里使用默认位置) ---
                        loc = env.unwrapped.unrealcv.get_obj_location(agent_name)
                        with unreal_lock:
                            env.unwrapped.unrealcv.nav_to_goal_bypath(obs_name, loc[:3])

                        # 阶段 1: 录制第一个动作 (QA Agent 初始化视频)
                        print(f"[Main-Thread] 正在录制第一阶段 (初始化视频)...")
                        video_frames_phase_1 = []
                        
                        # 循环直到收到 snapshot 
                        while not snapshot_event.is_set():
                            with unreal_lock:
                                obs = env.unwrapped.unrealcv.read_image(cam_id, 'lit', mode='direct')
                            if obs is not None:
                                # 修复颜色通道 (RGB -> BGR)
                                obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
                                video_frames_phase_1.append(obs)
                            time.sleep(0.05) 
                        
                        print(f"[Main-Thread] 第一阶段录制结束，捕获帧数: {len(video_frames_phase_1)}")
                        with unreal_lock:
                            env.unwrapped.unrealcv.set_pause()
                        
                        # 阶段 2: 保存环境快照 (Environment Reproduction)
                        print(f"[Main-Thread] 正在保存环境状态快照...")
                        env_snapshot = save_environment_snapshot(unwrapped_env, unreal_lock)
                        
                        # 保存快照到文件
                        agent_trace_dir = os.path.join(trace_dir, f"agent_{agent_name}_{trace_cnt}")
                        if not os.path.exists(agent_trace_dir): os.makedirs(agent_trace_dir)

                        snapshot_path = os.path.join(agent_trace_dir, f"snapshot_after_action1.json")
                        with open(snapshot_path, 'w') as f:
                            json.dump(env_snapshot, f, indent=4)
                        print(f"[Main-Thread] 快照已保存: {snapshot_path}")

                        with unreal_lock:   
                            env.unwrapped.unrealcv.set_resume()
                        # 阶段 3: 录制剩余动作 (意图理解视频)
                        print(f"[Main-Thread] 通知线程继续，开始录制第二阶段 (意图视频)...")
                        resume_event.set() # 通知线程继续
                        
                        video_frames_phase_2 = []
                        # 循环直到线程结束
                        while animation_thread.is_alive():
                            with unreal_lock:
                                obs = env.unwrapped.unrealcv.read_image(cam_id, 'lit', mode='direct')
                            if obs is not None:
                                # 修复颜色通道 (RGB -> BGR)
                                obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
                                video_frames_phase_2.append(obs)
                            time.sleep(0.05)
                        
                        print(f"[Main-Thread] 第二阶段录制结束，捕获帧数: {len(video_frames_phase_2)}")
                        
                        # --- 阶段 4: 保存视频 ---
                        # 视频 1: 给 QA Agent 用的 (仅包含第一个动作)
                        save_data({}, video_frames_phase_1, os.path.join(agent_trace_dir, "video_init"))
                        
                        # 视频 2: 完整的意图视频 (拼接)
                        full_video = video_frames_phase_1 + video_frames_phase_2
                        save_data({}, full_video, os.path.join(agent_trace_dir, "video_full_intent"))

                        # 清理
                        stop_event.set()
                        animation_thread.join()
                        print(f"[Main-Thread] {agent_name} 处理完毕。\n")
                        trace_cnt += 1
                    
                    # make dir
                    os.makedirs(current_gt_info, exist_ok=True)
                    os.makedirs(current_instance_obs_path, exist_ok=True)    # save obs

                    # save obs (这里保存的是最开始 cam_position 遍历的静态图，如果不需要可以注释掉)
                    saved_image_paths_for_instance = []
                    if collected_images_for_instance:
                        print(f"  Saving {len(collected_images_for_instance)} images for Instance ID: {instance_id} in Batch: {batch}...")
                        for img_data, original_cam_idx in collected_images_for_instance:
                            image_filename = os.path.join(current_instance_obs_path, f"obs_{original_cam_idx}.png")
                            try:
                                cv2.imwrite(image_filename, img_data) 
                                saved_image_paths_for_instance.append(image_filename)
                            except Exception as e:
                                print(f"    Error saving image {image_filename}: {e}")
                    
                    # save gt info
                    data_to_save = {
                        "instance_id": instance_id,
                        "safe_start": safe_start_to_collect,
                        "obs_folder_path": current_instance_obs_path,
                        "obs_filenames": [os.path.basename(p) for p in saved_image_paths_for_instance],
                        "sample_configs": info["sample_configs"],
                        "target_configs": current_target_configs,
                        "gt_information": info["gt_information"]
                    }
                    try:
                        with open(gt_info_filename, 'w') as f:
                            json.dump(data_to_save, f, indent=4) 
                        print(f"  Ground truth for Instance ID {instance_id} (Batch {batch}) saved to {gt_info_filename}")
                        
                        # update record file
                        new_entry = {instance_id: False}  
                        update_json_file(record_file, new_entry)
                        save_cnt += 1
                    except Exception as e:
                        print(f"  Error saving JSON for Instance ID {instance_id} (Batch {batch}): {e}")
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




