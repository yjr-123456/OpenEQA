import math
import numpy as np
import cv2

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
    car_loca = car_position[:3]  # [x, y, z]
    cat_rot = car_rotation[3:]   # [roll, pitch, yaw]
    
    theta = np.deg2rad(cat_rot[1])  # yaw角度转弧度
    bias = [250*np.cos(theta+np.pi/2), 250*np.sin(theta+np.pi/2), 0]
    
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

def obs_transform(obs):
    obs_rgb = cv2.cvtColor(obs[0][..., :3], cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    obs_depth = obs[0][..., -1]
    return obs_rgb, obs_depth
