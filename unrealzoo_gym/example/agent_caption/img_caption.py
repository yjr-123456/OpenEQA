from openai import OpenAI
import numpy as np
import json
import cv2
import os # 确保导入 os 模块
from dotenv import load_dotenv
load_dotenv(override=True)  

def load_model_config(config_path="model_config.json"):
    """加载模型配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"模型配置文件 {config_path} 不存在，使用默认配置")
        return None
    except json.JSONDecodeError:
        print(f"模型配置文件 {config_path} 格式错误")
        return None


def create_client(model_name="doubao", config_path="model_config.json"):
    """根据模型名称创建OpenAI客户端"""
    config = load_model_config(config_path)
    
    if config is None:
        raise ValueError(f"无法加载模型配置文件 {config_path}")
    
    model_config = config["models"].get(model_name)
    if model_config is None:
        print(f"模型 {model_name} 配置不存在，使用默认模型")
        model_name = config["default_model"]
        model_config = config["models"][model_name]
    key_name = model_config["api_key_env"]
    # 从环境变量获取API密钥
    api_key = os.environ.get(key_name)
    if not api_key:
        raise ValueError(f"环境变量 {key_name} 未设置")

    client = OpenAI(
        base_url=model_config["base_url"],
        api_key=api_key
    )
    
    return client, model_config["model_name"], model_config


client = None
current_model_name = None
current_model_config = None

def initialize_model(model_name="doubao",model_config_path="model_config.json"):
    
    global client, current_model_name, current_model_config
    client, current_model_name, current_model_config = create_client(model_name,model_config_path)
    print(f"Initialize model: {model_name} ({current_model_name})")



# system_prompt_player = """
#     We provide you with the front view, side view and back view of an object.
#     Create a description of the main, distinct object with concise, up to 30 words. 
#     Highlight its appearance, gender, skin color,and what he/she is wearing.
#     Do not return content with a period.
#     [output format]:
#     Gender: man\nBald:
# """
system_prompt_player = """
    We provide you with the front view, side view and back view of human.
    Create a description of the main, distinct object with concise. 
    Highlight its gender, skin color, bald or not, and whether he/she is wearing glasses.
    [Attention]:
    1. Do not return content with a period.
    2. You should describe the human in the form of [output format] that I provide to you. 
    [output format]:
    Gender: man\nSkin_color: black\nBald_or_not: false\nWearing_glasses_or_not: true\n
"""

system_prompt_animal = """
    We provide you with the front view, side view and back view of an animal.
    Create a description of the main, distinct object with concise, up to 20 words. 
    Highlight its species, appearance.
    Do not return content with a period.
"""

system_prompt_vehicle = """
    We provide you with the front view, side view and back view of a vehicle.
    Create a description of the main, distinct object with concise, up to 20 words. 
    Highlight its appearance,color,shape,style and structure.
    Do not return content with a period.
"""

system_prompt_obj = """
    We provide you with the front view, side view and top-down view of an object.
    Create a description of the main, distinct object with concise, up to 20 words. 
    Highlight its category, appearance,color and shape.
    Do not return content with a period.
"""

def parse_caption_to_dict(caption_string):
    """
    Parses a multi-line key-value string (like the GPT output format) into a dictionary.
    Converts 'true'/'false' strings to boolean values.
    """
    attributes = {}
    if not caption_string or not isinstance(caption_string, str):
        return attributes # Return empty dict if input is invalid

    lines = caption_string.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(':', 1) # Split only on the first colon
        if len(parts) == 2:
            key = parts[0].strip()
            value_str = parts[1].strip()
            
            # Convert boolean strings to actual booleans
            if value_str.lower() == 'true':
                attributes[key] = True
            elif value_str.lower() == 'false':
                attributes[key] = False
            else:
                attributes[key] = value_str
        else:
            print(f"Warning: Could not parse line: '{line}'")
    return attributes


def encode_image_array(image_array):
        from PIL import Image
        import io
        import base64
        # 检查图像是否已是 BGR 格式，如果不是，尝试转换
        # OpenAI 的 GPT-4o 通常期望 RGB 格式的图像。
        # cv2.imread 默认读取为 BGR。
        if image_array.ndim == 3 and image_array.shape[2] == 3: # 检查是否为3通道图像
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        if image_array.max() <= 1 and image_array.dtype == np.float32: # 检查是否为0-1范围的浮点数
            image_array = (image_array * 255).clip(0, 255)
        image_array = image_array.astype(np.uint8)
    
        img = Image.fromarray(image_array)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='png')
        base64_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        #print(base64_image[:50])
        return base64_image

def image_captioning(image0,image1,image2):
    # client = OpenAI(
    #         api_key="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"
    # )
    base64_image0 = encode_image_array(image0)
    base64_image1 = encode_image_array(image1)
    base64_image2 = encode_image_array(image2)
    response = client.chat.completions.create(
        model=current_model_name, # 确保模型名称正确
        max_tokens=10000,
        messages=[
            {"role": "system", "content": system_prompt_obj},
            {"role": "user", "content": [
                {
                "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image0}"}
                },
                {
                "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image1}"}
                },
                {
                "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image2}"}
                }
            ]
        }
        ],
    )
    respon = response.choices[0].message.content
    print(f"Generated caption: {respon}")
    return respon


if __name__ == "__main__":
    all_agent_types_data = {
        "player": [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16, 17,19],
        "animal": [0, 1, 2, 3, 6, 10, 11, 12, 14, 15, 16, 19, 20, 21, 22, 23, 25, 26, 27],
        "animal":[16],
        "drone": [0],
        "car": ["BP_Hatchback_child_base_C","BP_Hatchback_child_extras_C","BP_Hatchback_child_police_C","BP_Hatchback_child_taxi_C","BP_Sedan_child_base_C","BP_Sedan_child_extras_C","BP_Sedan_child_police_C","BP_Sedan_child_taxi_C"],
        "motorbike": [0,1,2,3,4,5,6,7,8],
        "pickup_items": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    }
    #agent_type = "animal"
    agent_type = "pickup_items"
    generated_features_data = {} # 用于存储所有生成的字幕

    # 选择要处理的 agent_type，或者可以遍历 all_agent_types_data.keys() 来处理所有类型
    # agent_types_to_process = ["player", "animal", "drone", "car", "motorbike"]
    agent_types_to_process = ["pickup_items"] # 例如，只处理 player 类型
    initialize_model("gemini_2.5_pro", "E:/EQA/unrealzoo_gym/example/solution/model_config.json")
    for agent_type in agent_types_to_process:
        if agent_type not in all_agent_types_data:
            print(f"Agent type {agent_type} not defined in all_agent_types_data. Skipping.")
            continue

        agent_ids_for_type = all_agent_types_data[agent_type]
        generated_features_data[agent_type] = {} # 为当前 agent_type 初始化一个字典

        print(f"\nProcessing agent type: {agent_type}")
        for id_val in agent_ids_for_type:
            img_path_base = f"./agent_render/{agent_type}/{id_val}"
            img0_path = f"{img_path_base}/0.png"
            img1_path = f"{img_path_base}/1.png"
            img2_path = f"{img_path_base}/2.png"

            # 检查图像文件是否存在
            if not (os.path.exists(img0_path) and os.path.exists(img1_path) and os.path.exists(img2_path)):
                print(f"Warning: One or more images not found for {agent_type} ID {id_val} in {img_path_base}. Skipping.")
                continue
            
            print(f"  Loading images for ID: {id_val} from {img_path_base}")
            img0 = cv2.imread(img0_path)
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            # 检查图像是否成功加载
            if img0 is None or img1 is None or img2 is None:
                print(f"Error: Failed to load one or more images for {agent_type} ID {id_val}. Skipping.")
                continue
            
            try:
                # image_captioning now returns a dictionary
                feature_dict = image_captioning(img0, img1, img2) 
                generated_features_data[agent_type][id_val] = feature_dict
                print(f"  Stored features for ID {id_val}: {feature_dict}")
                # exit(-1) # Removed exit(-1) to process all specified items
            except Exception as e:
                print(f"Error generating/parsing features for {agent_type} ID {id_val}: {e}")
                generated_features_data[agent_type][id_val] = {"error": f"ERROR_PROCESSING:_{e}"}


    output_dir = "./agent_caption/pickup_items" 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    output_json_path = os.path.join(output_dir,"agent_features.json") # Changed filename
    
    # 如果文件已存在，并且您想追加或合并，需要先读取现有数据
    existing_data = {}
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, 'r') as f_read:
                existing_data = json.load(f_read)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode existing JSON from {output_json_path}. Will overwrite.")
        except Exception as e:
            print(f"Error reading existing JSON file {output_json_path}: {e}. Will overwrite.")

    # 合并新生成的数据和现有数据
    # 简单的合并策略：新数据覆盖旧数据中相同 agent_type 和 id_val 的条目
    for agent_type, id_data in generated_features_data.items():
        if agent_type not in existing_data:
            existing_data[agent_type] = {}
        for id_val, features in id_data.items():
            existing_data[agent_type][id_val] = features


    try:
        with open(output_json_path, 'w') as f_write: # Use 'w' to write (potentially merged) data
            json.dump(existing_data, f_write, indent=4)
        print(f"\nSuccessfully saved generated features to {output_json_path}")
    except Exception as e:
        print(f"Error saving features to JSON: {e}")

    print("\nFinal generated_features_data (current run):")
    print(json.dumps(generated_features_data, indent=4))
    print("\nData saved in JSON file (merged):")
    print(json.dumps(existing_data, indent=4))



