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
    # create base directory with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base_dir = os.path.join(trace_dir, timestamp)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    # save data to json file
    with open(os.path.join(base_dir, 'data.json'), 'w') as f:
        json.dump(data, f, indent=4)
    # save images
    obs_dir = os.path.join(base_dir, 'obs')
    if not os.path.exists(obs_dir):
        os.makedirs(obs_dir)
    for i, img in enumerate(img_list):
        cv2.imwrite(os.path.join(obs_dir, f'image_{i}.png'), img)
    print(f"Data saved to {base_dir}")

def run_single_animation_in_background(unwrapped_env, agent_name, target_name, animation_type, stop_event):
    """
    在后台线程中为单个角色运行动画。
    这个函数会循环播放动画，直到主线程发出停止信号。
    """
    print(f"[Thread-Animation] 动画线程已为 {agent_name} 启动，动画: {animation_type}。")
    
    # 标记动画是否已执行，对于非循环动画只执行一次
    animation_done = False

    while not stop_event.is_set():
        # 对于只执行一次的复杂动画（如进车），执行后就等待
        if animation_done:
            time.sleep(0.5) # 线程继续运行但不再执行动作，等待停止信号
            continue

        print(f"[Thread-Animation] 正在为 {agent_name} 设置动画: {animation_type}")
        if animation_type == 'pick_up':
            loca = unwrapped_env.unrealcv.get_obj_location(agent_name)
            rot = unwrapped_env.unrealcv.get_obj_rotation(agent_name)
            theta = np.deg2rad(rot[1])
            bias = [50*np.cos(theta-np.pi/2), 50*np.sin(theta-np.pi/2), 0]
            loc = [loca[i] + bias[i] for i in range(3)]
            unwrapped_env.unrealcv.new_obj(pick_up_class, target_name, loc, rot)
            unwrapped_env.unrealcv.set_obj_color(target_name, np.random.randint(0, 255, 3))
            time.sleep(1)
            unwrapped_env.unrealcv.set_animation(agent_name, animation_type)
            animation_done = True # 标记为已完成

        elif animation_type == 'in_vehicle':
            if target_name:
                vehicle = target_name
                loca = unwrapped_env.unrealcv.get_obj_location(vehicle)
                rot = unwrapped_env.unrealcv.get_obj_rotation(vehicle)
                theta = np.deg2rad(rot[1])
                bias = [200*np.cos(theta+np.pi/2), 200*np.sin(theta+np.pi/2), 0]
                loc = [loca[i] + bias[i] for i in range(3)]
                unwrapped_env.unrealcv.set_obj_location(agent_name, loc)
                time.sleep(0.5)
                unwrapped_env.unrealcv.set_obj_rotation(agent_name, rot)
                time.sleep(0.5)
                unwrapped_env.unrealcv.set_animation(agent_name, 'enter_vehicle')
                animation_done = True # 标记为已完成
        else:
            # 对于其他简单循环动画，可以直接设置
            unwrapped_env.unrealcv.set_animation(agent_name, animation_type)
            animation_done = True # 如果也只执行一次，则标记
        time.sleep(1) 
    
    print(f"[Thread-Animation] 动画线程已为 {agent_name} 结束。")



if __name__ == '__main__':
    # env name
    parser = argparse.ArgumentParser(description=None)
    # parser.add_argument("-e", "--env_name", nargs='?', default="Map_ChemicalPlant_1", help='Select the environment name to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=0, help='random seed')
    parser.add_argument("-t", '--time-dilation', dest='time_dilation', default=10, help='time_dilation to keep fps in simulator')
    parser.add_argument("-n", '--nav-agent', dest='nav_agent', action='store_true', help='use nav agent to control the agents')
    parser.add_argument("-d", '--early-done', dest='early_done', default=-1, help='early_done when lost in n steps')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')
    parser.add_argument("-c", '--ifonly-counting', dest='if_cnt', action='store_true', help='if only counting the number of agents in the scene')
    # parser.add_argument("-g", "--reachable-points-graph", dest='graph_path', default=f"./agent_configs_sampler/points_graph/{env_name}/environment_graph_1.gpickle", help='use reachable points graph')
    # parser.add_argument("-w", "--work-dir", dest='work_dir', default=current_dir, help='work directory to save the data')
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
        # "Map_ChemicalPlant_1":{ 
        #     "height": -12776.0,
        #     "agent_categories": ['player','drone','car','animal'],
        #     },
        "SuburbNeighborhood_Day":{
            "height": 0,
            "agent_categories": ['player','drone','car', "robotdog"]
        },
        # "Pyramid": {
        #     "height": 0,
        #     "agent_categories": ['player','drone','animal']
        # },
        # "Greek_Island": {
        #     "height": 1258,
        #     "agent_categories": ['player','drone','animal']
        # },
        # "ModularNeighborhood": {
        #     "height": 136,
        #     "agent_categories": ['player','drone','car','animal']
        # },
        # "StonePineForest": {
        #     "height": -3065,
        #     "agent_categories": ['player','drone','car','animal']
        # },
        # "AsianMedivalCity": {
        #     "height": -1365,
        #     "agent_categories": ['player','drone','animal']
        # },
        # "LV_Bazaar": {
        #     "height": 138.0,
        #     "agent_categories": ['player','drone','animal']
        # },
        # "DowntownWest": {
        #     "height": 0,
        #     "agent_categories": ['player','drone','car','animal']
        # },
        # "PlanetOutDoor": {
        #     "height": 0,
        #     "agent_categories": ['player','drone']
        # },
        #         "RussianWinterTownDemo01": {
        #     "height": 0,
        #     "agent_categories": ['player','drone','car','animal']       
        # },
        # "Arctic": {
        #     "agent_categories": ['player','drone','animal']       
        # },
        # "Medieval_Castle": {
        #     "agent_categories": ['player','drone','animal']
        # },
        # "SnowMap": {
        #     "agent_categories": ['player','drone','animal']
        # },
        # "Real_Landscape": {
        #     "agent_categories": ['player','drone','animal',"robotdog"]
        # },
        # "Demonstration_Castle": {
        #     "agent_categories": ['player','drone','animal']
        # },
        # "Venice": {
        #     "agent_categories": ['player','drone','animal',"robotdog"]
        # },
    }

    all_type_ranges = {
        'player': (2, 4),
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
            # floor_height = env_dict[env_name]["height"]
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
                    for cam_idx, position in enumerate(cam_position):
                        loca = position[0:3]
                        rota = position[3:]
                        # place observer
                        env.unwrapped.unrealcv.set_obj_location(obs_name, loca)
                        env.unwrapped.unrealcv.set_obj_rotation(obs_name, rota)
                        time.sleep(0.1)  # wait for the observer to be placed
                        actions = action * agent_num
                        obs, _, _, _, _ = env.step(actions)
                        obs = obs[0]  # get the first observation
                        bgr_image_from_env = obs[...,:3].squeeze()
                        mask_obs = obs[...,3:].squeeze()
                        if bgr_image_from_env.dtype != np.uint8:
                            if bgr_image_from_env.max() <= 1.0 and bgr_image_from_env.min() >= 0.0:
                                bgr_image_from_env = (bgr_image_from_env * 255).astype(np.uint8)
                            else:
                                bgr_image_from_env = bgr_image_from_env.astype(np.uint8)
                        # set safe start
                        safe_start_to_collect.append(position)
                        print(f"    Image from camera {cam_idx + 1} marked for saving.")
                        collected_images_for_instance.append((bgr_image_from_env, cam_idx))
                    # obj animation and recording
                    for agent_name, animation_type in zip(player_info['name'], player_info['animation']):
                        print(f"\n[Main-Loop] 开始为角色 '{agent_name}' (动画: {animation_type}) 进行录制...")
                        # 1. 为当前角色创建一个停止事件和动画线程
                        if animation_type in ['pick_up', 'in_vehicle']:                            
                            # 主线程不等动画线程，立即开始录制
                            print(f"[Main-Thread] 主线程开始为 '{agent_name}' 移动观察者并录制...")
                            # collect trajectory
                            if animation_type == 'pick_up':
                                agent_type = 'player'
                                batch_id = agent_name.split('_')[-1]
                                pick_up_class = "BP_GrabMoveDrop_C"
                                pick_up_name = f"{pick_up_class}_{agent_name}" 
                                unwrapped_env.pickup_list.append(pick_up_name)
                                target_name = pick_up_name
                                name_index = current_target_configs[agent_type]['name'].index(agent_name)
                            elif animation_type == 'in_vehicle':
                                agent_type = 'car'
                                has_vehicle = any(unwrapped_env.agents[obj]['agent_type'] in ['car', 'motorbike'] for obj in unwrapped_env.target_list)
                                vehicle_list = [obj for obj in unwrapped_env.player_list if has_vehicle and unwrapped_env.agents[obj]['agent_type'] in ['car', 'motorbike']]   
                                if len(vehicle_list) > 0:
                                    obj_loc = unwrapped_env.unrealcv.get_obj_location(agent_name)
                                    min_distance = float('inf')
                                    nearest_vehicle = None
                                    for vehicle_name in vehicle_list:
                                        vehicle_loc = unwrapped_env.unrealcv.get_obj_location(vehicle_name)
                                        distance = calculate_distance(obj_loc, vehicle_loc)
                                        if distance < min_distance:
                                            min_distance = distance
                                            nearest_vehicle = vehicle_name
                                    target_name = nearest_vehicle
                                    name_index = current_target_configs[agent_type]['name'].index(target_name)
                                else:
                                    raise ValueError("No vehicle found for 'in_vehicle' animation.")
                            
                            # start animation thread
                            stop_event = threading.Event()
                            animation_thread = threading.Thread(
                                target=run_single_animation_in_background,
                                args=(unwrapped_env, agent_name, target_name, animation_type, stop_event)
                            )    
                            animation_thread.start()
                            loc = current_target_configs[agent_type]['start_pos'][name_index]
                            cur_location = position[:3] 
                            # collect video frames
                            img_list = []
                            pose_list = []
                            action_list = []
                            time_list = []
                            count_step = 0
                            path_string =env.unwrapped.unrealcv.nav_to_goal_bypath(obs_name,loc[:3])
                            print("=========path_string=========\n",path_string)
                            while check_reach(loc, cur_location):
                                obs=env.unwrapped.unrealcv.read_image(cam_id, 'lit', mode='direct')
                                cur_pose = env.unwrapped.unrealcv.get_obj_pose(obs_name)
                                time_list.append(time.time())
                                cur_location = cur_pose[:3]
                                img_list.append(obs)
                                pose_list.append(cur_pose)
                                count_step += 1
                            # caculate action for each frame
                            for i in range(0, len(pose_list)-1):
                                delta_yaw = pose_list[i+1][-2] - pose_list[i][-2]
                                direct_vec = np.array(pose_list[i+1][:2]) - np.array(pose_list[i][:2])
                                delta_t = time_list[i+1] - time_list[i]
                                distance = np.linalg.norm(direct_vec)
                                linear_velocity = distance / delta_t
                                angle_velocity = delta_yaw / delta_t
                                action_list.append([angle_velocity, linear_velocity])
                            data_to_save = {}
                            data_to_save["Target_Type"] = agent_type
                            data_to_save["Start_Pose"] = position
                            data_to_save["Target_Name"] = agent_name
                            data_to_save["Target_Pose"] = loc
                            data_to_save["Trajectory_Pose"] = pose_list
                            data_to_save["Action_Per_Frame"] = action_list
                            data_to_save["Time_Per_Frame"] = time_list
                            save_data(data_to_save, img_list,f"{trace_dir}/{trace_cnt}")
                            trace_cnt+=1
                    
                    # make dir
                    os.makedirs(current_gt_info, exist_ok=True)
                    os.makedirs(current_instance_obs_path, exist_ok=True)    # save obs

                    # save obs
                    saved_image_paths_for_instance = []
                    if collected_images_for_instance:
                        print(f"  Saving {len(collected_images_for_instance)} images for Instance ID: {instance_id} in Batch: {batch}...")
                        for img_data, original_cam_idx in collected_images_for_instance:
                            image_filename = os.path.join(current_instance_obs_path, f"obs_{original_cam_idx}.png")
                            try:
                                cv2.imwrite(image_filename, img_data) 
                                saved_image_paths_for_instance.append(image_filename)
                                print(f"    Saved: {image_filename}")
                            except Exception as e:
                                print(f"    Error saving image {image_filename}: {e}")
                    else:
                        print(f"  No images selected for saving for Instance ID: {instance_id} in Batch: {batch}.")
                    
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
                    print("gt_information\n",data_to_save)
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




