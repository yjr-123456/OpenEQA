import argparse
from torch.utils.tensorboard.summary import draw_boxes
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
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, configUE, augmentation
import os
import json
from example.solution.baseline.VLM.agent_predict import initialize_model, call_api_vlm
# from VLM_Agent.Rough_agent import agent
from ultralytics import YOLOWorld
import torch
import base64
import math
# from trajectory_visualizer import TrajectoryVisualizer
from dotenv import load_dotenv

load_dotenv()
from pynput import keyboard

def load_json_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading {file_path}: {e}")
        return None


def obs_transform(obs):
    obs = obs[0][..., :3]  
    return cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

key_state = {
    'i': False,
    'j': False,
    'k': False,
    'l': False,
    'w': False,
    'a': False,
    's': False,
    'd': False,
    'y': False,
    'c': False,
    'n': False,
    'b': False,
    'm': False,
    'n': False,
    'z': False,
    'x': False,
    'p': False,
    'space': False,
    'head_up': False,
    'head_down': False
}

def on_press(key):
    try:
        if key.char in key_state:
            key_state[key.char] = True
    except AttributeError:
        if key == keyboard.Key.space:
            key_state['space'] = True
        if key == keyboard.Key.up:
            key_state['head_up'] = True
        if key == keyboard.Key.down:
            key_state['head_down'] = True


def on_release(key):
    try:
        if key.char in key_state:
            key_state[key.char] = False
    except AttributeError:
        if key == keyboard.Key.space:
            key_state['space'] = False
        if key == keyboard.Key.up:
            key_state['head_up'] = False
        if key == keyboard.Key.down:
            key_state['head_down'] = False

def get_key_action():
    action= [6]
    # action = ([0, 0], 0, 0)
    # action = list(action)  # Convert tuple to list for modification
    # action[0] = list(action[0])  # Convert inner tuple to list for modification
    if key_state['i']:
        action = [0]
    if key_state['k']:
        action= [1]
    if key_state['j']:
        action = [2]
    if key_state['l']:
        action = [3]
    if key_state['z']:
        action = [5]
    if key_state['x']:
        action = [4]
    return action

def get_key_collection():
    collection = 0
    if key_state['p']:
        collection = 1
    return collection


import json
from datetime import datetime

def calculate_distance(pos1, pos2):
    """计算两个位置之间的欧几里得距离"""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)

def find_nearest_car(player_pos, car_positions):
    """找到离player最近的car"""
    min_distance = float('inf')
    nearest_car_idx = -1
    
    for i, car_pos in enumerate(car_positions):
        distance = calculate_distance(player_pos, car_pos)
        if distance < min_distance:
            min_distance = distance
            nearest_car_idx = i
    
    return nearest_car_idx, min_distance

def adjust_player_position_near_car(car_position, car_rotation):
    """
    根据car的位置和旋转角度计算player的新位置
    将player放置在car旁边250单位距离处
    """
    car_loca = car_position[:3]  # [x, y, z]
    cat_rot = car_rotation[3:]   # [roll, pitch, yaw]
    
    theta = np.deg2rad(cat_rot[1])  # yaw角度转弧度
    bias = [200*np.cos(theta+np.pi/2), 200*np.sin(theta+np.pi/2), 0]
    
    new_position = [
        car_loca[0] + bias[0],
        car_loca[1] + bias[1], 
        car_loca[2] + bias[2],
        car_position[3],  
        car_position[4],  # 保持原来的pitch
        car_position[5]   # 保持原来的yaw
    ]
    
    return new_position

def process_in_vehicle_players(config_data):
    """
    处理配置数据，调整in_vehicle状态的player位置
    """
    target_configs = config_data["target_configs"]
    
    # 检查是否有player和car数据
    if "player" not in target_configs or "car" not in target_configs:
        print("Warning: Missing player or car data in configuration")
        return config_data
    
    players = target_configs["player"]
    cars = target_configs["car"]
    
    # 获取player数据
    player_animations = players.get("animation", [])
    player_positions = players.get("start_pos", [])
    player_names = players.get("name", [])
    
    # 获取car数据
    car_positions = cars.get("start_pos", [])
    car_names = cars.get("name", [])
    
    print(f"Found {len(player_animations)} players and {len(car_positions)} cars")
    
    # 检查每个player的animation
    for i, animation in enumerate(player_animations):
        if animation == "in_vehicle":
            player_name = player_names[i] if i < len(player_names) else f"player_{i}"
            player_pos = player_positions[i] if i < len(player_positions) else None
            
            if player_pos is None:
                print(f"Warning: No position data for player {player_name}")
                continue
                
            if not car_positions:
                print(f"Warning: No cars available for player {player_name}")
                continue
            
            print(f"\nProcessing player {player_name} with 'in_vehicle' animation")
            print(f"Original player position: {player_pos}")
            
            # 找到最近的car
            nearest_car_idx, distance = find_nearest_car(player_pos, car_positions)
            nearest_car_name = car_names[nearest_car_idx] if nearest_car_idx < len(car_names) else f"car_{nearest_car_idx}"
            nearest_car_pos = car_positions[nearest_car_idx]
            
            print(f"Nearest car: {nearest_car_name} at {nearest_car_pos}")
            print(f"Distance: {distance:.2f} units")
            
            # 计算新的player位置
            new_player_pos = adjust_player_position_near_car(nearest_car_pos, nearest_car_pos)
            
            print(f"New player position: {new_player_pos}")
            
            # 更新配置数据
            target_configs["player"]["start_pos"][i] = new_player_pos
            
            print(f" Updated position for player {player_name}")
    
    return config_data

def depth_to_base64(depth_array):
    """将深度数组转换为16位PNG的base64编码，保留更高精度"""
    import io
    from PIL import Image
    import cv2
    
    # 保存原始信息用于恢复
    original_min = depth_array.min()
    original_max = depth_array.max()
    
    print(f"Original depth range: [{original_min:.6f}, {original_max:.6f}]")
    
    # 转换为16位（65536个层级 vs 8位的256个层级）
    if original_max > original_min:
        depth_16bit = ((depth_array - original_min) / 
                      (original_max - original_min) * 65535).astype(np.uint16)
    else:
        depth_16bit = np.zeros_like(depth_array, dtype=np.uint16)
    
    # 使用OpenCV保存16位PNG
    buffered = io.BytesIO()
    
    # 将16位数组编码为PNG字节
    success, encoded_img = cv2.imencode('.png', depth_16bit)
    if success:
        buffered.write(encoded_img.tobytes())
        depth_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        raise ValueError("Failed to encode 16-bit depth image")
    
    # 返回base64和恢复信息
    metadata = {
        'original_min': float(original_min),
        'original_max': float(original_max),
        'bit_depth': 16,
        'encoding': '16bit_png'
    }
    
    return depth_base64, metadata



listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()


client = None
current_model_name = None
current_model_config = None
sys_prompt = """
    You are a helpful assistant.We will ask you a question. Given the depth image, You should compare the distances of objects from your view.
"""
user_prompt = """
    Based on the depth image, which objects are closer to you, the house or the people?
"""


if __name__ == '__main__':
    env_list = [
        "SuburbNeighborhood_Day",
        # "ModularNeighborhood",
        # "ModularSciFiVillage", 
        # "Cabin_Lake",
        # "Pyramid",
        # "RuralAustralia_Example_01",
        # "ModularVictorianCity",
        # "Map_ChemicalPlant_1"
    ]
    try:
        for env_name in env_list:
            question_type = None
            save_dir = None
            total_questions = 0
            correct_answers = 0
            results = []
            results_filename = None
            parser = argparse.ArgumentParser()
            parser.add_argument("-e", "--env_id", default=f'UnrealCvEQA_general-{env_name}-DiscreteRgbd-v0')
            parser.add_argument("-s", "--seed", type=int, default=0)
            parser.add_argument("-t", "--time-dilation", type=int, default=-1)
            parser.add_argument("-d", "--early-done", type=int, default=-1)
            parser.add_argument("-p", "--QA_path", default=os.path.join(os.path.dirname(__file__), 'GT_info'))
            parser.add_argument("--resume", action='store_true', help="Resume from previous progress")
            parser.add_argument("--model", default="doubao", help="choose evaluation models")
            parser.add_argument("--config_path", default="E:\\EQA\\unrealzoo_gym\\example\\solution", help="configuration file path")
            parser.add_argument("--save_dir", default="E:\\EQA\\unrealzoo_gym\\example\\depth_image", help="observation depth")
            args = parser.parse_args()
            # init agent
            obs_name = "BP_Character_C_1"
            print("Initializing UnrealCV Gym environment...")
            os.makedirs(args.save_dir, exist_ok=True)
            initialize_model(args.model, f"{args.config_path}/model_config.json")            
            env_dir = f"{args.QA_path}/{env_name}"

            scenario_folder_names = [d for d in os.listdir(env_dir) 
                                        if os.path.isdir(os.path.join(env_dir, d))]
            status_file = json.load(open(f"{env_dir}/status_recorder.json", 'r'))
            for scenario_folder_name in [scenario_folder_names[0]]:
                if status_file.get(scenario_folder_name, False) == False:
                    print(f"Skipping already processed scenario: {scenario_folder_name}")
                    continue
                # register env
                env = gym.make(args.env_id)
                if args.time_dilation > 0:
                    env = time_dilation.TimeDilationWrapper(env, args.time_dilation)
                if args.early_done > 0:
                    env = early_done.EarlyDoneWrapper(env, args.early_done)
    
                id_folder_path = os.path.join(env_dir, scenario_folder_name)
                file_path = os.path.join(id_folder_path, f"gt_info.json")
                    
                if not os.path.isfile(file_path):
                    print(f"Warning: JSON file not found in {id_folder_path}, skipping.")
                    continue
                        
                # 加载QA数据
                QA_data_loaded = load_json_file(file_path)
                if QA_data_loaded is None:
                    continue
                        
                        # 处理in_vehicle状态的player位置
                temp_config_data = {"target_configs": QA_data_loaded.get("target_configs", {})}
                processed_config_data = process_in_vehicle_players(temp_config_data)
                QA_data_loaded["target_configs"] = processed_config_data["target_configs"]
                        
                print(f"\n--- Processing File for Interaction: {file_path} ---")
                
                # 环境设置代码
                target_configs = QA_data_loaded.get("target_configs", {})
                safe_start_config = QA_data_loaded.get("safe_start")
                if len(safe_start_config) > 1:
                    safe_start_config = [safe_start_config]
                refer_agents_category_config = QA_data_loaded.get("refer_agents_category")
                agent_num = sum(len(target_configs.get(t, {}).get("name", [])) for t in target_configs)
                start_pose = QA_data_loaded.get("safe_start", [])
                if len(start_pose) == 1:
                    start_pose = start_pose[0]
                assert len(start_pose) == 6
                        
                unwrapped_env = env.unwrapped
                unwrapped_env.safe_start = safe_start_config
                unwrapped_env.target_configs = target_configs
                unwrapped_env.refer_agents_category = list(target_configs.keys())
                env = augmentation.RandomPopulationWrapper(env, num_min=agent_num + 1, num_max=agent_num + 1)
                env = configUE.ConfigUEWrapper(env, resolution=(512,512), offscreen=False)

                print(f"Resetting environment for file: {os.path.basename(file_path)}")
                states, info = env.reset()
                obs_rgb = states[0][..., :3].squeeze()
                obs_depth = states[0][...,-1].squeeze()
                obs = cv2.cvtColor(obs_rgb, cv2.COLOR_BGR2RGB)
                # convert depth image
                print(f"Depth data shape: {obs_depth.shape}, range: [{obs_depth.min():.3f}, {obs_depth.max():.3f}], type: {obs_depth.dtype}")
                print("=================Camera Config =================")
                env.unwrapped.unrealcv.get_camera_config()
                # 将深度数据转换为base64
                depth_base64, metadata = depth_to_base64(obs_depth)

                # 调用API时传递base64字符串而不是NumPy数组
                respon = call_api_vlm( sys_prompt, user_prompt, [depth_base64])
                print(f"Model Response: {respon}")
                img_path = os.path.join(args.save_dir, f"{args.model}", "img")
                os.makedirs(img_path, exist_ok=True)
                np.save(os.path.join(img_path, f"{env_name}_depth.npy"), obs_depth)
                cv2.imwrite(os.path.join(img_path, f"{env_name}_rgb.png"), obs_rgb)
                response_data = {
                    'env_name': env_name,
                    'scenario': scenario_folder_name,
                    'timestamp': datetime.now().isoformat(),
                    'api_response': respon,
                    'depth_shape': obs_depth.shape,
                    'depth_metadata': metadata
                }
                respon_path = os.path.join(args.save_dir, f"{args.model}", "model_response")
                os.makedirs(respon_path, exist_ok=True)
                # 保存为JSON
                response_json_path = os.path.join(respon_path, f"{env_name}_{args.model}_response.json")
                with open(response_json_path, 'w', encoding='utf-8') as f:
                    json.dump(response_data, f, indent=2, ensure_ascii=False)
                print(f"Response JSON saved to: {response_json_path}")
                # collection_choice = 0
                # key_state['p'] = False
                # while collection_choice ==0:
                #     collection_choice = get_key_collection()
                #     time.sleep(0.1)
                # if collection_choice == 1:
                #     status_file[scenario_folder_name] = False
                #     with open(f"{env_dir}/status_recorder.json", 'w') as f:
                #         json.dump(status_file, f, indent=4)
                # # env.close()
                cv2.destroyAllWindows()
                status_file[scenario_folder_name] = False
                with open(f"{env_dir}/status_recorder.json", 'w') as f:
                    json.dump(status_file, f, indent=4)
                env.close()

    except KeyboardInterrupt:
        print("Process interrupted by user. Exiting...")
        if env:
            env.close()
    finally:
        if env:
            env.close()
        print("Environment closed.")
        cv2.destroyAllWindows()
        listener.stop()

