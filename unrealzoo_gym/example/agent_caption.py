import os
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import cv2 # Import cv2 first
import numpy as np

import argparse
#import gym_unrealcv
import matplotlib
matplotlib.use('Agg')
import gymnasium as gym
# from gymnasium import wrappers

import time
import numpy as np
import os
# import torch
#from gym_unrealcv.envs.tracking.baseline import PoseTracker
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, agents, augmentation, configUE
import random
from openai import OpenAI
import json
from pynput import keyboard

scene_prompt = (
    "There is a 3D virtual scene with objects in it. The scene contains a supermarket with a variety of objects.Please describe the scene as much detail as possible."
)

prompt_bg_rela_loca = (
    "There is a dictionary containing ground truth informations of objects in the 3D virtual scene"
    "You should ask **relative location questions about the objects** in the scene according to the dictionary."
    "Make sure the question has four options and a definite and true answer."
    "Try to be creative in asking question and make it sound like interesting scenarios.\n"
    "[Question Type]:\n"
    "The question type should be relative locations between objects, such as 'Where is the woman relative to the man'.\n"
    "Please make sure that every question has four options containing one definite and true answer\n"
    "[Some Tips]:\n"
    "1. Make full use of the ground truth relative position information provided in the dictionary.That means, if there are n objects in the scene, the dictionary will provide you n*(n-1)/2 relative position information, and you can ask n*(n-1)/2 questions.(A's relative position to B and B's relative position to A are seen as one question).Please ask as many questions as you can.\n"
    "2. As it is difficult for agent to figure out the measurement in 3D virtual environment, you can only ask questions in a more imprecise way like ' Where is the man1 relative to the woman1?\n A. to the front-right of\n B. to the front-left of\n C. to the back-right of\n D. to the back-left of\n'.\n"
    "3. You can't ask duplicate questions like 'Where is man1 relative to woman1' and 'where is woman1 relative to man1'.Because they have same meaning, and the second question is meaningless.\n"
    "4. We will provide you an image caption containing objects in the scene, maybe you can use it to generate interesting questions.\n"
    "[Output Format]:\n" 
    "Question 1: Where is the man relative to the woman?\n\n A. To the front of\n B. behind\n C. To the left of\n D. To the right of\n\n"
    "Answer 1: B. To the front of\n\n"
)

prompt_bg_rela_distance = (
    "There is a dictionary containing ground truth informations of objects in the 3D virtual scene"
    "You should ask **relative location questions about the objects** in the scene according to the dictionary."
    "Make sure the question has four options and a definite and true answer."
    "Try to be creative in asking question and make it sound like interesting scenarios.\n"
    "[Question Type]:\n"
    "The question type should be relative locations between objects, such as 'Where is the woman relative to the man'.\n"
    "Please make sure that every question has four options containing one definite and true answer\n"
    "[Some Tips]:\n"
    "1. Make full use of the ground truth relative position information provided in the dictionary.That means, if there are n objects in the scene, the dictionary will provide you n*(n-1)/2 relative position information, and you can ask n*(n-1)/2 questions.(A's relative position to B and B's relative position to A are seen as one question).Please ask as many questions as you can.\n"
    "2. As it is difficult for agent to figure out the measurement in 3D virtual environment, you can only ask questions in a more imprecise way like ' Where is the man1 relative to the woman1?\n A. to the front-right of\n B. to the front-left of\n C. to the back-right of\n D. to the back-left of\n'.\n"
    "3. You can't ask duplicate questions like 'Where is man1 relative to woman1' and 'where is woman1 relative to man1'.Because they have same meaning, and the second question is meaningless.\n"
    "4. We will provide you an image caption containing objects in the scene, maybe you can use it to generate interesting questions.\n"
    "[Output Format]:\n" 
    "Question 1: How many man in the scene?\n A. 1\n B. 2\n C. 3\n D. 4\n\n"
    "Answer 1: A. 1\n\n"
)

prompt_bg_state = (
    "There is a dictionary containing ground truth informations of objects in the 3D virtual scene"
    "You should ask **objects' states question** in the scene according to the dictionary."
    "Make sure the question has four options and a definite and true answer."
    "Try to be creative in asking question and make it sound like interesting scenarios.\n"
    "[Question Type]:\n"
    "The question type should be **objects'states question**, such as 'What's the state of the man'.\n"
    "Please make sure that every question has four options containing one definite and true answer\n"
    "[Some Tips]:\n"
    "1. Make full use of the ground truth state information provided in the dictionary.That means, if there are n objects in the scene, then you can ask n questions for all objects mentioned in the dictionary.Please ask as many questions as you can\n"
    "2. We will provide you an image caption containing objects in the scene, maybe you can use it to generate interesting questions.\n"
    "3. Do not give away state information in your question stem like 'What's the state of the man in blue plaid pants lying on the suburban street?'!That is meaningless."
    "[Output Format]:\n" 
    "Question 1: What's the state of the man?\n\n A. lying down\n B. standing\n C. crouch\n D. others\n\n"
    "Answer 1: A. lying down\n\n"
)

prompt_bg_cnt = (
    "There is a dictionary containing ground truth informations of objects in the 3D virtual scene"
    "You should ask **object counting questions about the objects** in the scene according to the dictionary."
    "Make sure the question has four options and a definite and true answer."
    "Try to be creative in asking question and make it sound like interesting scenarios.\n"
    "[Question Type]:\n"
    "The question type should be **objects counting**, such as 'How many man in the scene'.\n"
    "Please make sure that every question has four options containing one definite and true answer\n"
    "[Some Tips]:\n"
    "1. Make full use of the ground truth information provided in the dictionary.\n"
    "2. You can summarize some sharing features of objects such as 'bald man', 'wether wearing glasses or not', and ask object counting question via these features.For example, 'How many bald mans in the scene?Please ask as many questions as you can'\n"
    "3. We will provide you an image caption containing objects in the scene, maybe you can use it to generate interesting questions.\n"
    "3. Counting numbers must be expressed in English, such as 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'.\n"
    "[Output Format]:\n" 
    "Question 1: How many man in the scene?\n\n A. one \n B. two\n C. three\n D. four\n\n"
    "Answer 1: A. one\n\n"
)

filter_prompt = (
    "There are serval questions generated by the LLMs, but they may repeat each other.And some of have wrong answer\n"
    "Please filter the questions and keep the ones that have correct answer and are not repeated.\n"
    "We will provide you the dictionary which contains the ground truth information of the objects in the questions.You can use it to check the correctness of the answers to the questions.\n"
    "[notions]:\n"
    "1. The questions are genrated in three times, and different questions might have same serial numbers, so please rearrange the questions first.\n"
    "1. Duplicate questions: questions that have same meaning, like 'Where is man1 relative to woman1' and 'where is woman1 relative to man1'."
    "2. Question Type: the question type should be relative location between objects."
    "3. How to check the correct answer of the questions: you can use the dictionary to check the correctness of the answer to the questions.If the answers are wrong, please correct the answers(Maybe you need to change the options if the ground truth answer is not contained in the options)\n"
    "4. Please rearrange the questions in order, and make sure the question number is continuous.\n"
    "[output format]:"
    "Question 1: The question you ask, please make sure they have four options\n"
    "Answer 1: The answer to the question, please make sure that the answer is true\n"
)

def format_dict_for_llm(enhanced_dict, question_type = "relative_location"):

    import numpy as np
    
    result = "\n"
    
    def format_array(arr):
        if isinstance(arr, np.ndarray):
            arr = arr.tolist()
        return str(arr)
    
    for obj_name, obj_info in enhanced_dict.items():
        result += f"Object: {obj_name}\n"
        result += f"  State: {obj_info['state']}\n"
        result += f"  Location: {format_array(obj_info['location'])}\n"
        result += f"  Rotation: {format_array(obj_info['rotation'])}\n"
        if question_type == 'relative_location' and 'relative_positions_description' in obj_info:
            result += "  Relative positions:\n"
            for other_obj, rel_info in obj_info['relative_positions_description'].items():
                result += f"    To {other_obj}: {rel_info['description']}\n"
        result += "\n"
          
    
    return result

def encode_image_array(image_array):
        from PIL import Image
        import io
        import base64
        if image_array.max() <= 1:
            image_array = (image_array * 255).clip(0, 255)
        image_array = image_array.astype(np.uint8)
    
        img = Image.fromarray(image_array)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='png')
        base64_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        #print(base64_image[:50])
        return base64_image

def image_captioning(image):
    client = OpenAI(
            api_key=""
    )
    base64_image = encode_image_array(image)
    response = client.chat.completions.create(
        model='gpt-4o-2024-08-06',
        max_tokens=1000,
        messages=[
            {"role": "system", "content": "Please describe the scene below."},
            {"role": "user", "content": [
                {
                "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                }
            ]
        }
        ],
    )
    respon = response.choices[0].message.content
    print(respon)
    return respon

def generate_question(dictionary,image_description, question_type = "relative_location"):    

    client = OpenAI(
            api_key="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"
    )
    prompt = ""
    if question_type == "relative_location":
        prompt = prompt_bg_rela_loca
    elif question_type == "counting":
        prompt = prompt_bg_cnt
    elif question_type == "state":
        prompt = prompt_bg_state
    else:
        raise ValueError("Invalid question type. Please choose 'relative_location', 'counting', or 'state'.")
        

    response = client.chat.completions.create(
        model='gpt-4o-2024-08-06',
        max_tokens=10000,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {
                    "type": "text",
                    "text": f"scenary description:{image_description}"
                },
                {
                    "type":"text",
                    "text": f"dictionary:{dictionary},the dictionary contains the ground truth information of the objects in the scene"
                },
                {
                    "type": "text",
                    "text": f"Please ask **object {question_type}** questions about the objects in the scene according to the dictionary and scene description."
                }
                ]
            },
        ],

    )
    respon = response.choices[0].message.content
    print(respon)
    return respon

def filter_questions(questions, dictionary):
    client = OpenAI(
            api_key="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"
    )
    response = client.chat.completions.create(
        model='gpt-4o-2024-08-06',
        max_tokens=10000,
        messages=[
            {"role": "system", "content": filter_prompt},
            {"role": "user", "content": [
                {
                    "type": "text",
                    "text": f"questions:{questions}"
                },
                {
                    "type":"text",
                    "text": f"dictionary:{dictionary}"
                },
                {
                    "type": "text",
                    "text": "Please filter the questions and keep the ones that have correct answer and are not repeated,and rearrange them in order."
                }
                ]
            },
        ],

    )
    respon = response.choices[0].message.content
    print(respon)
    return respon

def save_collection_data_to_single_file(qa_string, image_list,target_configs, refer_agents_category, safe_start,
                                        env_name="SuburbNeighborhood_Day", question_type="relative_location", 
                                        batch_id=None, base_dir="./datacollection"):
    """
    Save data to a single JSON file with multiple batches
    
    Args:
        qa_string: Question-answer text
        target_configs: Target configuration dictionary
        refer_agents_category: Reference agent category list
        safe_start: Safe start position list
        env_name: Environment name
        question_type: Question type
        batch_id: Optional batch ID, if None, timestamp will be used
        base_dir: Base directory
    """
    import re
    import json
    import os
    import time
    
    # Generate batch ID
    if batch_id is None:
        batch_id = f"batch_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # Create save directory
    save_dir = os.path.join(base_dir, env_name)
    save_dir = os.path.join(save_dir, question_type)
    os.makedirs(save_dir, exist_ok=True)
    
    # Define file path (single file for all batches)
    data_file = os.path.join(save_dir, f"{question_type}.json")
    
    # Save observations
    obs_folder = f"obs_{batch_id}"
    obs_dir = os.path.join(save_dir, obs_folder)
    os.makedirs(obs_dir, exist_ok=True)
    for i, obs in enumerate(image_list):
        cv2.imwrite(os.path.join(obs_dir, f"obs_{i}.png"), obs)

    # Parse QA data
    questions_dict = {}
    pattern = r'Question (\d+): (.*?)\n\n((?:\s*[A-D]\.\s.*?\n)+)\n*Answer \1: ([A-D]\..*?)(?:\n\n|$)'
    #pattern = r'Question (\d+): (.*?)(?:\n((?:[A-D]\. .*?\n)+))Answer \1: ([A-D]\. .*?)(?:\n\n|$)'
    matches = re.finditer(pattern, qa_string, re.DOTALL)
    



    for match in matches:
        q_num = match.group(1)
        question_text = match.group(2).strip()
        options_text = match.group(3).strip()
        answer_text = match.group(4).strip()
        
        options = []
        for option_line in options_text.split('\n'):
            if option_line.strip():
                options.append(option_line.strip())
        
        questions_dict[f"Question {q_num}"] = {
            "question": question_text,
            "options": options,
            "answer": answer_text
        }
    
    # Create batch data
    filtered_target_configs = {}
    for category in refer_agents_category:
        if category in target_configs:
            filtered_target_configs[category] = target_configs[category]

    batch_data = {
        "target_configs": filtered_target_configs, 
        "refer_agents_category": refer_agents_category,
        "safe_start": safe_start,
        "obs_dir": obs_dir,
        "Questions": questions_dict
    }
    
    # Read existing file or create new structure
    if os.path.exists(data_file) and os.path.getsize(data_file) > 0:
        try:
            with open(data_file, 'r') as f:
                file_data = json.load(f)
            
            # Add new batch to existing file
            if "batches" not in file_data:
                file_data = {"batches": {}}
            
        except json.JSONDecodeError:
            print(f"Warning: {data_file} is not valid JSON. Creating new file.")
            file_data = {"batches": {}}
    else:
        # Create new file structure
        file_data = {"batches": {}}
    
    # Add the new batch
    file_data["batches"][batch_id] = batch_data
    
    # Save to file
    with open(data_file, 'w') as f:
        json.dump(file_data, f, indent=4)
    
    print(f"Batch {batch_id} saved to {data_file}")
    
    # Update index file (for backward compatibility)
    index_file = os.path.join(save_dir, "sessions_index.json")
    if os.path.exists(index_file):
        with open(index_file, 'r') as f:
            try:
                index_data = json.load(f)
            except:
                index_data = {"sessions": []}
    else:
        index_data = {"sessions": []}
    
    # Add batch info to index
    index_data["sessions"].append({
        "session_id": batch_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "file": os.path.basename(data_file),
        "question_count": len(questions_dict),
        "question_type": question_type,
        "batch_id": batch_id
    })
    
    # Save updated index
    with open(index_file, 'w') as f:
        json.dump(index_data, f, indent=4)
    
    return batch_id

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space
        self.count_steps = 0
        self.action = self.action_space.sample()

    def act(self, observation, keep_steps=10):
        self.count_steps += 1
        if self.count_steps > keep_steps:
            self.action = self.action_space.sample()
            self.count_steps = 0
        else:
            return self.action
        return self.action

    def reset(self):
        self.action = self.action_space.sample()
        self.count_steps = 0


#use keyboard
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
    if key_state['y']:
        collection = 1
    elif key_state['m']:
        collection = 2
    elif key_state['b']:
        collection = 3
    return collection


# def extract_agent_from_image(color_img, mask_img):
#     import numpy as np
#     import cv2
    
#     hsv_mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2HSV)
#     yellow_lower = np.array([20, 100, 100])  
#     yellow_upper = np.array([40, 255, 255])  
#     binary_mask = cv2.inRange(hsv_mask, yellow_lower, yellow_upper)
    
#     rgba_output = np.zeros((color_img.shape[0], color_img.shape[1], 4), dtype=np.uint8)
    

#     rgba_output[:,:,:3] = color_img
    
    
#     rgba_output[:,:,3] = binary_mask
    
#     bg_removed = color_img.copy()
#     bg_removed[binary_mask == 0] = [0, 0, 0]  
    
#     return rgba_output, bg_removed



def extract_agent_from_image(color_img, mask_img, crop_to_bbox=True):
    import numpy as np
    import cv2
    
    # 1. 通过 mask_img 中的黄色分割出 binary_mask
    hsv_mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2HSV)
    yellow_lower = np.array([28, 250, 250])  
    yellow_upper = np.array([32, 255, 255])  
    binary_mask = cv2.inRange(hsv_mask, yellow_lower, yellow_upper) # 单通道 HxW 掩码
    
    # 2. 创建完整的 RGBA 图像和背景移除的图像
    # rgba_output_full: color_img 加上来自 binary_mask 的 alpha 通道
    # 首先将 color_img (假设是BGR) 转换为BGRA，初始alpha为255
    rgba_output_full = cv2.cvtColor(color_img, cv2.COLOR_BGR2BGRA)
    rgba_output_full[:,:,3] = binary_mask # 用二值掩码设置alpha通道 (对象处为255, 其他为0)
    
    # bg_removed_full: color_img，但非对象区域变为白色
    bg_removed_full = color_img.copy()
    bg_removed_full[binary_mask == 0] = [255, 255, 255] # <--- 修改此处，将背景设为白色
    
    if not crop_to_bbox:
        return rgba_output_full, bg_removed_full

    # 3. 从 binary_mask 计算边界框 (类似于 get_bbox(normalize=False) 的逻辑)
    pixel_points = cv2.findNonZero(binary_mask)

    if pixel_points is not None and len(pixel_points) > 0:
        # pixel_points 是 (N, 1, 2) 形状的数组, 每个点是 [[x, y]]
        x_coordinates = pixel_points[:, :, 0]
        y_coordinates = pixel_points[:, :, 1]
        
        x_min = np.min(x_coordinates)
        x_max = np.max(x_coordinates)
        y_min = np.min(y_coordinates)
        y_max = np.max(y_coordinates)
        
        # 确保边界框有效 (至少1x1像素)
        if x_max >= x_min and y_max >= y_min:
            # 使用计算得到的边界框裁剪图像
            # NumPy 切片 [y_start:y_end, x_start:x_end]，其中 y_end 和 x_end 是不包含的
            cropped_rgba_output = rgba_output_full[y_min : y_max + 1, x_min : x_max + 1]
            cropped_bg_removed = bg_removed_full[y_min : y_max + 1, x_min : x_max + 1]
            
            return cropped_rgba_output, cropped_bg_removed
        else:
            # 边界框退化 (例如，一条线或一个点)，返回完整图像作为后备
            # print("警告: 从binary_mask得到的边界框退化，返回完整图像。")
            return rgba_output_full, bg_removed_full
    else:
        # 在 binary_mask 中未找到代理 (binary_mask 全为零)
        # 返回原始的 (可能是全黑或全透明的) 完整图像
        # print("警告: 在binary_mask中未找到用于裁剪的代理，返回完整图像。")
        return rgba_output_full, bg_removed_full





if __name__ == '__main__':
    #from .eqa_agent import Agent
    env_name = "Map_ChemicalPlant_1"
    agent_type = "motorbike"
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env_id", nargs='?', default=f'UnrealCvEQA_general-{env_name}-DiscreteColorMask-v4',
                        help='Select the environment to run')
    # parser.add_argument("-e", "--env_id", nargs='?', default=f'UnrealCvEQA_general-{env_name}-DiscreteColorMask-v0',
    #                     help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=0, help='random seed')
    parser.add_argument("-t", '--time-dilation', dest='time_dilation', default=-1, help='time_dilation to keep fps in simulator')
    parser.add_argument("-n", '--nav-agent', dest='nav_agent', action='store_true', help='use nav agent to control the agents')
    parser.add_argument("-d", '--early-done', dest='early_done', default=-1, help='early_done when lost in n steps')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')
    
    args = parser.parse_args()
    
    env = gym.make(args.env_id)

    # agents category
    env.unwrapped.refer_agents_category=[agent_type]
    car_have_done = ["BP_Hatchback_child_base_C","BP_Hatchback_child_extras_C","BP_Hatchback_child_police_C","BP_Hatchback_child_taxi_C","BP_Sedan_child_base_C","BP_Sedan_child_extras_C","BP_Sedan_child_police_C","BP_Sedan_child_taxi_C"]
    Vehicles={ #only available in latest UE5.5 package
       "car":[
            "BP_SUV_child_base_C","BP_SUV_child_extras_C","BP_SUV_child_police_C","BP_SUV_child_taxi_C"],
    "motorbike":["BP_BaseBike_C", "BP_Custom_Base_C","BP_Custom_Extras_C","BP_Custom_Police_C"
                ,"BP_Enduro_Base_C","BP_Enduro_Extras_C","BP_Enduro_Police_C"
                  ,"BP_Naked_Base_C","BP_Naked_Extras_C","BP_Naked_Police_C"
                  ,"BP_BaseBike_TwoPassengers_C"]
}
    # get unwrapper infomation:
    unwrapped_env = env.unwrapped
    # target_configs = unwrapped_env.target_config
    # refer_agents_category = unwrapped_env.refer_agents_category
    # safe_start = unwrapped_env.safe_start #start point when testing
    # env wrappers
    if int(args.time_dilation) > 0:  # -1 means no time_dilation
        env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
    if int(args.early_done) > 0:  # -1 means no early_done
        env = early_done.EarlyDoneWrapper(env, int(args.early_done))
    if args.monitor:
        env = monitor.DisplayWrapper(env)
    # env = augmentation.RandomPopulationWrapper(env, 2,2, random_target=False)
    env = configUE.ConfigUEWrapper(env, offscreen=False,resolution=(1080,1080))

    episode_count = 1
    reward = 0
    done = False
    # listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    # listener.start()

    #id condigs
    # player_id = [21]
    # animal_id2 = [0, 1, 2, 3, 12, 19, 25, 26, 27]
    # animal_id = [6,10,11,14,15, 20,21,22,23,24]
    # drone_id = [0]
    # car_id = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    # motorbike_id = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

    # player cam_id
    cam_id = env.unwrapped.cam_list[0]
    action = [-1,-1]  # stay still
    obs, info = env.reset()
    valid_name = "BP_Hatchback_child_base_C_1"  # default valid name for car
    refer_agent_category = ['car', 'motorbike']  # default refer agent category
    try:
        for agent_class in Vehicles[agent_type]:
            if agent_type == "car":
                valid_name = "BP_Hatchback_child_base_C_1"
            elif agent_type == "motorbike":
                valid_name = "BP_BaseBike_C_1"
            # env.unwrapped.unrealcv.new_obj(agent_class, f"{agent_type}_1", [-19609.924, -11846.606,-12792.063], [0, 0, 0])
            refer_agent = env.unwrapped.refer_agents[valid_name]
            refer_agent["class_name"] = agent_class 
            env.unwrapped.target_start.append([-19609.924, -11846.606,-12792.063,0, 0, 0])
            env.unwrapped.agents[f"{agent_type}_1"] = env.unwrapped.add_agent(f"{agent_type}_1",[-19609.924, -11846.606,-12792.063,0, 0, 0], refer_agent)
            save_directory = f"./agent_render/{agent_type}/{agent_class}"
            os.makedirs(save_directory, exist_ok=True)

            env.unwrapped.unrealcv.cam = env.unwrapped.unrealcv.get_camera_config()
            env.unwrapped.update_camera_assignments()

            image_list_for_saving = [] 
            # set appearance
            # env.unwrapped.unrealcv.set_appearance(f"{agent_type}_1", app_id)
            env.unwrapped.unrealcv.set_obj_rotation(f"{agent_type}_1", [0, 0, 0])
            time.sleep(1)
            obs,_,_,_,_ = env.step(action)
            print(obs.shape)
            obs_color = obs[0][...,:3].squeeze()
            obs_mask_for_segmentation = obs[0][...,3:].squeeze()
            cv2.imshow("obs_color", obs_color)
            cv2.imshow("obs_mask_for_segmentation", obs_mask_for_segmentation)
            cv2.waitKey(1)
            
            cropped_agent_rgba, cropped_agent_on_white = extract_agent_from_image(
                obs_color, 
                obs_mask_for_segmentation, 
                crop_to_bbox=True
            )
            cv2.imshow("cropped_agent_on_white", cropped_agent_on_white)
            cv2.waitKey(1)
            if cropped_agent_on_white is not None and cropped_agent_on_white.size > 0:
                image_list_for_saving.append(cropped_agent_on_white)
            
            # turn the agent by 90 degrees
            # env.unwrapped.unrealcv.set_obj_location("BP_Character_2", [-850,200, 200])
            env.unwrapped.unrealcv.set_obj_rotation(f"{agent_type}_1", [0, 90, 0])
            time.sleep(1)
            obs,_,_,_,_ = env.step(action)
            obs_color = obs[0][...,:3].squeeze()
            obs_mask_for_segmentation = obs[0][...,3:].squeeze()
            cv2.imshow("obs_color", obs_color)
            cv2.imshow("obs_mask_for_segmentation", obs_mask_for_segmentation)
            cv2.waitKey(1)
            cropped_agent_rgba, cropped_agent_on_white = extract_agent_from_image(
                obs_color, 
                obs_mask_for_segmentation, 
                crop_to_bbox=True
            )
            cv2.imshow("cropped_agent_on_white", cropped_agent_on_white)
            cv2.waitKey(1)
            if cropped_agent_on_white is not None and cropped_agent_on_white.size > 0:
                image_list_for_saving.append(cropped_agent_on_white)

            # turn the agent by 90 degrees
            # env.unwrapped.unrealcv.set_obj_location("BP_Character_2", [-9158,-1200.583,98.82])
            # time.sleep(2)
            env.unwrapped.unrealcv.set_obj_rotation(f"{agent_type}_1", [0, 180, 0])
            time.sleep(1)
            obs,_,_,_,_ = env.step(action)
            obs_color = obs[0][...,:3].squeeze()
            obs_mask_for_segmentation = obs[0][...,3:].squeeze()
            cv2.imshow("obs_color", obs_color)
            cv2.imshow("obs_mask_for_segmentation", obs_mask_for_segmentation)
            cv2.waitKey(1)
            cropped_agent_rgba, cropped_agent_on_white = extract_agent_from_image(
                obs_color, 
                obs_mask_for_segmentation, 
                crop_to_bbox=True
            )
            cv2.imshow("cropped_agent_on_white", cropped_agent_on_white)
            cv2.waitKey(1)
            if cropped_agent_on_white is not None and cropped_agent_on_white.size > 0:
                image_list_for_saving.append(cropped_agent_on_white)
            
            if len(image_list_for_saving) > 1:
                for i in range(len(image_list_for_saving)):
                    cv2.imwrite(os.path.join(save_directory, f"{i}.png"), image_list_for_saving[i])
            # env.unwrapped.unrealcv.destroy_pickup(f"{agent_type}_1")
            env.unwrapped.remove_agent(f"{agent_type}_1")
            
            
            # cnt_step = 0
            # t0 = time.time()
            # while True:
            #     action = [6]
            #     print("please get action!")
            #     while action == [6] and not any(key_state[k] for k in ['y', 'm', 'b']):
            #         action = get_key_action()
            #         time.sleep(0.1)
            #     if action != [6]:
            #         obs, reward, termination, truncation, info = env.step(action)
            #         obs_color = obs[...,:3].squeeze()
            #         obs_mask = obs[...,3:].squeeze()
            #         cnt_step += 1
            #         time.sleep(1)
            #         print("please press 'y' to save the image, 'm' to continue, 'b' to stop")
            #         collection = 0
            #         while collection == 0:
            #             collection = get_key_collection()
            #             time.sleep(0.1)
            #         if collection == 1:
            #             image.append(obs_color)
            #             print("image appended")
            #             time.sleep(0.1)
            #         elif collection ==2:
            #             print("continue move!")
            #             time.sleep(0.1)
            #             continue
            #         elif collection ==3:
            #             print("stop move!")
            #             time.sleep(0.1)    
            #             break
                   

            # for question_type in ['state','relative_location','counting']:
            #     image_rgb = cv2.cvtColor(image[-1], cv2.COLOR_BGR2RGB)
            #     image_description = image_captioning(image_rgb)
            
            #     questions = generate_question(format_dict_for_llm(obj_dict,question_type), image_description,question_type)
            #     print(questions)

                
            #     batch_id = save_collection_data_to_single_file(
            #             questions, 
            #             image,
            #             target_configs, 
            #             refer_agents_category, 
            #             safe_start,
            #             env_name=env_name,
            #             question_type=question_type
            #         )
            #     print(f"Questions have been saved to batch {batch_id}")


        env.close()
    except KeyboardInterrupt:
        env.close()
        print('exiting')
        
