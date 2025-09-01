import argparse
from torch.utils.tensorboard.summary import draw_boxes
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)  # Add parent directory to sys.path

import gym
from gym import wrappers
import cv2
import time
import numpy as np
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, configUE, augmentation
import os
import json
from example.solution.baseline.VLM.agent_predict import agent, setup_logging
# from VLM_Agent.Rough_agent import agent
from ultralytics import YOLOWorld
import torch
import base64
import logging
import math
# from trajectory_visualizer import TrajectoryVisualizer
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

client = OpenAI(
    # 此为默认路径，您可根据业务所在地域进`行配置
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
    api_key=os.environ.get("ARK_API_KEY"),
)

# client = OpenAI(api_key=api_key)
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
    obs_rgb = cv2.cvtColor(obs[0][..., :3], cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    obs_depth = obs[0][..., -1]
    return obs_rgb, obs_depth

class HybridAgent:
    def __init__(self,reference_text,reference_image):
        # Initialize VLM agent
        self.vlm_agent = agent(reference_text,reference_image)
        
        # Initialize detection model
        self.yolo_model = YOLOWorld('yolov8x-worldv2.pt')
        
        # State tracking
        # self.person_detected = False
        # self.stretcher_detected = False
        # self.current_target = None  # 'person' or 'stretcher'
        
        # Movement parameters
        # self.foreward = {
        #     'angular': 0,
        #     'velocity': 50,
        #     'viewport': 0,
        #     'interaction': 0,
        # }
        #
        # self.backward = {
        #     'angular': 0,
        #     'velocity': -50,
        #     'viewport': 0,
        #     'interaction': 0,
        # }
        # self.turnleft = {
        #     'angular': -20,
        #     'velocity': 0,
        #     'viewport': 0,
        #     'interaction': 0,
        # }
        # self.turnright = {
        #     'angular': 20,
        #     'velocity': 0,
        #     'viewport': 0,
        #     'interaction': 0,
        # }
        # self.carry = {
        #     'angular': 0,
        #     'velocity': 0,
        #     'viewport': 0,
        #     'interaction': 3,
        # }
        # self.drop = {
        #     'angular': 0,
        #     'velocity': 0,
        #     'viewport': 0,
        #     'interaction': 4,
        # }
        # self.noaction = {
        #     'angular': 0,
        #     'velocity': 0,
        #     'viewport': 0,
        #     'interaction': 0,
        # }

        self.foreward = ([0,50],0,0)
        self.backward = ([0,-50],0,0)
        self.turnleft = ([-20,0],0,0)
        self.turnright = ([20,0],0,0)
        self.carry = ([0,0],0,3)
        self.drop = ([0,0],0,4)
        self.noaction = ([0,0],0,0)
    def reset(self, question, obs, target_category):
        self.vlm_agent.reset(question, obs)
        self.yolo_model.set_classes(target_category)
        


    def draw_bbox_on_obs(self,obs, boxes, labels, color=(0, 255, 0), thickness=2):
        """
        Draw bounding boxes on the observation image.

        Args:
            obs: The observation image (numpy array).
            boxes: List of bounding boxes, each in the format [x, y, w, h].
            labels: List of labels corresponding to the bounding boxes.
            color: Color of the bounding box (default: green).
            thickness: Thickness of the bounding box lines (default: 2).
        """
        for box, label in zip(boxes, labels):
            x, y, w, h = box
            top_left = (int(x - w / 2), int(y - h / 2))
            bottom_right = (int(x + w / 2), int(y + h / 2))
            cv2.rectangle(obs, top_left, bottom_right, color, thickness)
            cv2.putText(obs, label, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return obs

    def predict(self, obs, info):
        # First try to detect objects using YOLO
        results = self.yolo_model.predict(source=obs)
        
        return self.vlm_agent.predict(results, info)

    def _move_based_on_detection(self, box, target_type):
        x0, y0, w_, h_ = box
        
        if target_type == 'person':
            # if w_ > h_:
            if y0 - 0.5*h_ > 350 and x0>220 and x0<420:
                # self.person_detected = True
                return self.carry
            else:
                if x0 < 220:
                    return self.turnleft
                elif x0 > 420:
                    return self.turnright
                else:
                    return self.foreward
        elif target_type =='stretcher':
            if y0 - 0.5*h_ > 350  and x0>220 and x0<420 :
                return self.drop
            else:
                if x0 < 220:
                    return self.turnleft
                elif x0 > 420:
                    return self.turnright
                else:
                    return self.foreward
        elif target_type == 'truck':  # stretcher
            if w_> 220 and h_>220  and x0>220 and x0<420 :
                return self.drop
            else:
                if x0 < 220:
                    return self.turnleft
                elif x0 > 420:
                    return self.turnright
                else:
                    return self.foreward

def compare_answers_with_api(agent_answer, ground_truth, question_stem="", question_type=""):

    if agent_answer is None or ground_truth is None:
        return False
    
    # 首先尝试简单的字符串匹配（快速判断明显相同的答案）
    agent_answer_str = str(agent_answer).strip()
    ground_truth_str = str(ground_truth).strip()
    
    if agent_answer_str.lower() == ground_truth_str.lower():
        return True
    
    # 如果简单匹配失败，调用API进行智能判断
    return call_api_for_answer_comparison(agent_answer_str, ground_truth_str, question_stem, question_type)

def call_api_for_answer_comparison(agent_answer, ground_truth, question_stem="", question_type=""):

    # 构建系统提示
    system_prompt = """You are an expert evaluator for question-answering tasks. Your job is to determine if an agent's answer equals to the ground truth answer.
        Return only "CORRECT" if the answers match or "INCORRECT" if they don't match.
        Do not provide explanations, just the verdict."""

    # 构建用户提示
    user_prompt = f"""Question Type: {question_type}
        Question: {question_stem}
        Ground Truth Answer: {ground_truth}
        Agent's Answer: {agent_answer}
        If the agent's answer is correct, return "CORRECT". If it is incorrect, return "INCORRECT".
        """

    try:

        response = client.chat.completions.create(
            model='doubao-seed-1-6-thinking-250615',  
            max_tokens=50,  
            temperature=0.3,  
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        result = response.choices[0].message.content.strip().upper()
        
        # 解析结果
        if "CORRECT" == result:
            return True
        elif "INCORRECT" == result:
            return False
        else:
            # 如果API返回了意外的格式，回退到原始方法
            print(f"Warning: Unexpected API response: {result}. Falling back to string comparison.")
            return fallback_compare_answers(agent_answer, ground_truth)
            
    except Exception as e:
        print(f"Error calling API for answer comparison: {e}")
        # API调用失败时回退到原始方法
        return fallback_compare_answers(agent_answer, ground_truth)

def fallback_compare_answers(agent_answer, ground_truth):
    """
    API调用失败时的回退方法（原始的字符串比较逻辑）
    """
    if agent_answer is None or ground_truth is None:
        return False
    
    # 转换为字符串并标准化
    agent_answer = str(agent_answer).strip().lower()
    ground_truth = str(ground_truth).strip().lower()
    
    # 直接比较
    if agent_answer == ground_truth:
        return True
    
    # 处理选择题情况（A、B、C、D）
    if len(agent_answer) == 1 and len(ground_truth) == 1:
        return agent_answer == ground_truth
    
    # 处理数字答案
    try:
        agent_num = float(agent_answer)
        truth_num = float(ground_truth)
        return abs(agent_num - truth_num) < 0.01  # 允许小的浮点误差
    except ValueError:
        pass
    
    # 检查关键词包含关系
    if agent_answer in ground_truth or ground_truth in agent_answer:
        return True
    return False

def calculate_accuracy(correct, total):
    """计算准确率"""
    if total == 0:
        return 0.0
    return (correct / total) * 100

import json
from datetime import datetime

def save_results_to_file(results, correct_answers, total_questions, env_name=None, question_type=None, filename_prefix="./experiment_results"):
    """
    保存结果到JSON文件
    """
    # 根据参数生成文件名
    if env_name and question_type:
        filename = f"{env_name}_{question_type}.json"
    else:
        # 回退到时间戳命名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.json"
    
    os.makedirs(filename_prefix, exist_ok=True)
    file_path = os.path.join(filename_prefix, filename)
    
    # 准备保存的数据
    save_data = {
        "env_name": env_name,
        "question_type": question_type,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total_questions": total_questions,
        "correct_answers": correct_answers,
        "accuracy": calculate_accuracy(correct_answers, total_questions),
        "detailed_results": [
            {
                "question_id": i + 1,
                "scenario_name": result[2] if len(result) > 2 else "unknown",  # scenario信息
                "agent_answer": result[0],
                "ground_truth": result[1]
            }
            for i, result in enumerate(results)
        ]
    }
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        print(f" Results saved to: {filename}")
        return filename
    except Exception as e:
        print(f" Error saving results: {e}")
        return None

def append_results_to_file(results, correct_answers, total_questions, filename, env_name=None, question_type=None, filename_prefix="./experiment_results"):
    """
    追加结果到现有文件
    """
    file_path = os.path.join(filename_prefix, filename)

    try:
        # 读取现有数据
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        else:
            existing_data = {
                "env_name": env_name,
                "question_type": question_type,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "total_questions": 0,
                "correct_answers": 0,
                "accuracy": 0.0,
                "detailed_results": []
            }
        
        # 更新数据
        existing_data["total_questions"] = total_questions
        existing_data["correct_answers"] = correct_answers
        existing_data["accuracy"] = calculate_accuracy(correct_answers, total_questions)
        existing_data["last_updated"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 添加新的详细结果
        start_idx = len(existing_data["detailed_results"])
        for i, result in enumerate(results[start_idx:], start=start_idx):
            existing_data["detailed_results"].append({
                "question_id": i + 1,
                "scenario_name": result[2] if len(result) > 2 else "unknown",  # scenario信息
                "agent_answer": result[0],
                "ground_truth": result[1]
            })
        
        # 保存更新后的数据
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)
        
        print(f" Results updated in: {filename}")
        return True
    except Exception as e:
        print(f" Error updating results: {e}")
        return False


def load_or_create_state_file(state_file_path):
    """
    加载或创建状态文件
    """
    if os.path.exists(state_file_path):
        try:
            with open(state_file_path, 'r') as f:
                state_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            state_data = {}
    else:
        state_data = {}
    
    return state_data

def update_state_file(state_file_path, scenario_name, question_id, status="completed", 
                     agent_answer=None, ground_truth=None, is_correct=None):
    """
    更新状态文件，包含答案和正确性信息
    """
    state_data = load_or_create_state_file(state_file_path)
    
    if scenario_name not in state_data:
        state_data[scenario_name] = {}
    
    state_data[scenario_name][question_id] = {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "agent_answer": agent_answer,
        "ground_truth": ground_truth,
        "is_correct": is_correct
    }
    
    try:
        with open(state_file_path, 'w') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error updating state file: {e}")
        return False

def is_question_completed(state_data, scenario_name, question_id):
    """
    检查问题是否已完成
    """
    if scenario_name not in state_data:
        return False
    
    if question_id not in state_data[scenario_name]:
        return False
    
    return state_data[scenario_name][question_id].get("status") == "completed"

def load_completed_results_from_state(state_data):
    """
    从状态文件中加载已完成问题的统计信息
    
    Returns:
        tuple: (total_completed_questions, correct_answers, results_list)
    """
    total_completed = 0
    correct_answers = 0
    results = []
    
    for scenario_name, scenario_data in state_data.items():
        if isinstance(scenario_data, dict):
            for question_id, question_info in scenario_data.items():
                if isinstance(question_info, dict) and question_info.get("status") == "completed":
                    total_completed += 1
                    
                    agent_answer = question_info.get("agent_answer")
                    ground_truth = question_info.get("ground_truth")
                    is_correct = question_info.get("is_correct", False)
                    
                    if is_correct:
                        correct_answers += 1
                    
                    # 添加scenario信息到结果列表中
                    results.append((agent_answer, ground_truth, scenario_name))
    
    return total_completed, correct_answers, results

def get_completed_stats_for_scenario(state_data, scenario_name, qa_dict):
    """
    获取特定场景的完成统计信息
    """
    if scenario_name not in state_data:
        return 0, 0, []
    
    scenario_data = state_data[scenario_name]
    if not isinstance(scenario_data, dict):
        return 0, 0, []
    
    completed_count = 0
    correct_count = 0
    scenario_results = []
    
    for question_id in qa_dict.keys():
        if question_id in scenario_data:
            question_info = scenario_data[question_id]
            if isinstance(question_info, dict) and question_info.get("status") == "completed":
                completed_count += 1
                
                agent_answer = question_info.get("agent_answer")
                ground_truth = question_info.get("ground_truth")
                is_correct = question_info.get("is_correct", False)
                
                if is_correct:
                    correct_count += 1
                
                scenario_results.append((agent_answer, ground_truth, scenario_name))
    
    return completed_count, correct_count, scenario_results

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




if __name__ == '__main__':
    env_list = [
        "ModularSciFiVillage", 
        "Cabin_Lake",
        "Pyramid",
        "RuralAustralia_Example_01",
        "ModularNeighborhood",
        "ModularVictorianCity",
        "Map_ChemicalPlant_1"
    ]
    question_type_folders = ["counting", "relative_location", "state", "relative_distance"]
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
            parser.add_argument("-p", "--QA_path", default=os.path.join(os.path.dirname(__file__), 'QA_Data'))
            parser.add_argument("--resume", action='store_true', help="Resume from previous progress")
            parser.add_argument("--model", default="doubao", help="choose evaluation models")
            parser.add_argument("--config_path", default="E:\\EQA\\unrealzoo_gym\\example\\solution", help="configuration file path")
            args = parser.parse_args()
            
            # init agent
            AG = agent(model = args.model, config_path=args.config_path)
            obs_name = "BP_Character_C_1"
            print("Initializing UnrealCV Gym environment...")

            for q_type_folder_name in question_type_folders:
                question_type = q_type_folder_name
                type_specific_folder_dir = os.path.join(args.QA_path, env_name, q_type_folder_name)
                if not os.path.isdir(type_specific_folder_dir):
                    print(f"Warning: Type folder not found {type_specific_folder_dir}, skipping.")
                    continue
                
                scenario_folder_names = [d for d in os.listdir(type_specific_folder_dir) 
                                       if os.path.isdir(os.path.join(type_specific_folder_dir, d))]
                
                # 状态文件
                state_file_path = os.path.join(type_specific_folder_dir, "status_recorder.json")
                state_data = load_or_create_state_file(state_file_path)
                
                # 初始化统计变量
                total_questions = 0
                correct_answers = 0
                results = []
                results_filename = None
                save_dir = f"experiment_results/{question_type}"
                os.makedirs(save_dir, exist_ok=True)
                
                if args.resume:
                    completed_questions, completed_correct, completed_results = load_completed_results_from_state(state_data)
                    total_questions = completed_questions
                    correct_answers = completed_correct
                    results = completed_results
                    print(f"Resume mode: Found {completed_questions} completed questions")
                    print(f"Resume mode: Previous correct answers: {completed_correct}")
                    print(f"Resume mode: Previous accuracy: {calculate_accuracy(completed_correct, completed_questions):.2f}%")
                
                for scenario_folder_name in scenario_folder_names:
                    env = gym.make(args.env_id)
                    if args.time_dilation > 0:
                        env = time_dilation.TimeDilationWrapper(env, args.time_dilation)
                    if args.early_done > 0:
                        env = early_done.EarlyDoneWrapper(env, args.early_done)
                    id_folder_path = os.path.join(type_specific_folder_dir, scenario_folder_name)
                    file_path = os.path.join(id_folder_path, f"{q_type_folder_name}.json")
                    
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
                    
                    QA_dict = QA_data_loaded.get("Questions", {})
                    if not QA_dict:
                        print(f"Warning: No questions found in {file_path}, skipping.")
                        continue
                    
                    # 检查该场景是否已完全完成
                    if args.resume and scenario_folder_name in state_data:
                        scenario_data = state_data[scenario_folder_name]
                        if isinstance(scenario_data, dict):
                            all_completed = all(is_question_completed(state_data, scenario_folder_name, qid) 
                                              for qid in QA_dict.keys())
                            if all_completed:
                                print(f"Scenario {scenario_folder_name} already completed, skipping.")
                                continue
                    
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
                    unwrapped_env.refer_agents_category = refer_agents_category_config
                    unwrapped_env.target_configs = target_configs
                    unwrapped_env.is_eval = True
                    
                    env = augmentation.RandomPopulationWrapper(env, num_min=agent_num + 1, num_max=agent_num + 1, height_bias=100)
                    
                    env = configUE.ConfigUEWrapper(env, resolution=(512,512), offscreen=False)

                    print(f"Resetting environment for file: {os.path.basename(file_path)}")
                    states, info = env.reset()
                    obs_rgb, obs_depth = obs_transform(states)
                    for question_id, question_data in QA_dict.items():
                        # 检查单个问题是否已完成
                        if args.resume and is_question_completed(state_data, scenario_folder_name, question_id):
                            print(f"Question {question_id} in {scenario_folder_name} already completed, skipping.")
                            continue
                        
                        total_questions += 1
                        
                        # 设置玩家位置
                        loca = start_pose[0:3]
                        rota = start_pose[3:]
                        env.unwrapped.unrealcv.set_obj_location(obs_name, loca)
                        env.unwrapped.unrealcv.set_obj_rotation(obs_name, rota)

                        question_stem = question_data.get("question", "")
                        question_options = question_data.get("options", [])
                        question_answer = question_data.get("answer", None)
                        
                        if not question_stem or not question_answer:
                            print(f"Warning: Question stem is not complete in {file_path}, skipping.")
                            continue
                        log_base_path = os.path.dirname(__file__)
                        AG.reset(question=question_stem, obs_rgb=obs_rgb, obs_depth=obs_depth, target_type=refer_agents_category_config,
                                question_type=question_type, answer_list=question_options, batch_id = scenario_folder_name,
                                question_answer=question_answer, env_name=env_name,logger_base_dir=log_base_path)

                        max_step = AG.max_step
                        answer = None
                        cur_step = 0
                        
                        for cur_step in range(0, max_step+1):
                            action = AG.predict(obs_rgb, obs_depth,info)
                            actions = action + [-1]*agent_num
                            print(actions)
                            obs, reward, termination, truncation, info = env.step(actions)
                            obs_rgb, obs_depth = obs_transform(obs)

                            if AG.termination:
                                answer = AG.final_answer
                                break
                            if AG.truncation:
                                answer = AG.final_answer
                                print(f"Episode truncated after {cur_step} steps.")
                                break
                        
                        # 判断答案正确性
                        is_correct = compare_answers_with_api(
                            agent_answer=answer, 
                            ground_truth=question_answer,
                            question_stem=question_stem,
                            question_type=question_type
                        )
                        
                        # 更新统计 - 添加scenario信息到results中
                        results.append((answer, question_answer, scenario_folder_name))  # 添加scenario_folder_name
                        if is_correct:
                            correct_answers += 1
                            print(f" Correct answer for question: {question_stem}")
                            print(f"  Expected: {question_answer}, Got: {answer}")
                        else:
                            print(f" Incorrect answer for question: {question_stem}")
                            print(f"  Expected: {question_answer}, Got: {answer}")
                        
                        # 更新状态文件，包含答案和正确性信息
                        update_state_file(
                            state_file_path, 
                            scenario_folder_name, 
                            question_id, 
                            status="completed",
                            agent_answer=answer,
                            ground_truth=question_answer,
                            is_correct=is_correct
                        )
                        
                        # 定期保存结果
                        if total_questions % 10 == 0:
                            current_accuracy = calculate_accuracy(correct_answers, total_questions)
                            print(f"\n === Progress Update ===")
                            print(f"Processed {total_questions} questions so far.")
                            print(f"Correct answers: {correct_answers}")
                            print(f"Current accuracy: {current_accuracy:.2f}%")
                            print("=" * 30)
                        
                        # 保存结果到文件
                        if results_filename is None:
                            results_filename = save_results_to_file(
                                results, correct_answers, total_questions,
                                env_name=env_name, question_type=question_type,
                                filename_prefix=save_dir
                            )
                        else:
                            append_results_to_file(
                                results, correct_answers, total_questions, results_filename,
                                env_name=env_name, question_type=question_type,
                                filename_prefix=save_dir
                            )
                        
                        print(f"Question {total_questions} | Accuracy: {calculate_accuracy(correct_answers, total_questions):.1f}%")
                    env.close()
                    print(f"Completed scenario: {scenario_folder_name} in environment: {env_name}")
                    time.sleep(10)
                # env.close()    
            # env.close()      
    except KeyboardInterrupt:
        if 'env' in locals():
            env.close()
        print("\n=== 程序被中断 ===")
        print(f"已处理 {total_questions} 个问题")
        print(f"正确答案: {correct_answers}")
        if total_questions > 0:
            print(f"当前准确率: {calculate_accuracy(correct_answers, total_questions):.2f}%")
        print("进度已保存，可以使用 --resume 参数继续")
    except Exception as e:
        print(f"An error occurred: {e}")
        if 'env' in locals():
            env.close()
            print("Environment closed due to error.")
        raise e  
    finally:
        if 'env' in locals():
            env.close()
            print("Environment closed.")
            
            if total_questions > 0 and question_type and save_dir:
                final_accuracy = calculate_accuracy(correct_answers, total_questions)
                print(f"\n === FINAL RESULTS ===")
                print(f"Total Questions: {total_questions}")
                print(f"Correct Answers: {correct_answers}")
                print(f"Final Accuracy: {final_accuracy:.2f}%")
                print("=" * 30)
                
                if results_filename is None:
                    results_filename = save_results_to_file(
                        results, correct_answers, total_questions,
                        env_name=env_name, question_type=question_type,
                        filename_prefix=save_dir
                    )
                else:
                    append_results_to_file(
                        results, correct_answers, total_questions, results_filename,
                        env_name=env_name, question_type=question_type,
                        filename_prefix=save_dir
                    )
                
                print(f" All results saved to: {results_filename}")




