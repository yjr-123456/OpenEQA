# import sys
# import argparse
#import gym_unrealcv
# import gym
# from gym import wrappers
import cv2
import time
import numpy as np
import os
# import torch
#from gym_unrealcv.envs.tracking.baseline import PoseTracker
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, agents, augmentation, configUE, sample_agent_configs
import random
current_dir = os.path.dirname(os.path.abspath(__file__))
from openai import OpenAI
import json
from dotenv import load_dotenv
load_dotenv(True)  # Load environment variables from .env file
import argparse

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


scene_prompt = (
    "There are serval observations in different views,please give a comprehensive of them in the form of a paragraph.\n"
    "Just describe the scene as if you are in the scene, and do not use words like'The series of images depicts ...'\n"
)

system_prompt = (
    "You are a creative quizzer."
    "Your task is to ask questions about the objects in the scene according to the ground truth information we provide to you."
)

prompt_bg_rela_loca = (
    "Your task is to ask **relative location questions about the objects** in the scene according to the ground truth information we provide to you."
    "This is a question about relative locations between two objects, such as 'Where is the Object A relative to the Object B?'."
    "[Ground Truth Information Format]:\n"
    "The ground truth information is organized in this way:\n"
    "Firstly,the object's basic information: name and it's state.\n"
    "Secondly, other objects' distances to the target object are provided closely behind\n"
    "Object: target_object_name, State: object state\n"
    "  The object A is in front of the target_object\n"
    "  The object B is behind the target_object\n"
    "  ..."
    "[Question Type]:\n"
    "The question type should be relative locations between objects, such as 'Where is the woman relative to the man'.\n"
    "Please make sure that every question has four options containing one definite and true answer\n"
    "Only at most five questions you can ask, please ask meaningful questions.\n"
    "[Think Step by Step]:\n"
    "You can do this step by step:\n"
    "1. figure out the target object, and pick the another one object(object A).\n"
    "2. Ask question in this way: 'Where is the object A relative to the target object?'"
    "3. Given ground truth information, find the relative location from target_object to object A.\n"
    "[Important Attention]:"
    "1. Make sure that the target object's state is not **'liedown' and 'Flying'**, becaues it is hard to describe the relative location of an object from a lying down or flying object.\n"
    "[Output Format]:\n" 
    "Question 1: Where is the man relative to the woman?\n\n A. To the front of\n B. behind\n C. To the left of\n D. To the right of\n\n"
    "Answer 1: B. To the front of\n\n"
)

prompt_bg_rela_distance =(
    "Your task is to ask **relative distance comparison questions about the objects** in the scene according to the ground truth information we provide to you."
    "This is a question about comparing which of two objects(object A or object B) is closer or farther to a certain object(target object)."
    "For example, 'which is closer to the target object, object A or object B?'."
    "[Ground Truth Information Format]:\n"
    "The ground truth information is organized in this way:\n"
    "Firstly,the target object's basic information: name and it's state.\n"
    "Secondly, other objects' distances to the target object are provided closely behind\n"
    "Object: target_object_name, State: object state\n"
    "  The distance between object A and target object is distance_A\n"
    "  The distance between object B and target object is distance_B\n"
    "  ...\n"
    "[Think Step by Step]:\n"
    "You can do this step by step:\n"
    "1. figure out the target object, and pick the other two objects(object A and object B).\n"
    "2. Make sure the distances of two picked objects(object A and object B) to the target object are clearly to tell(distance difference at least greater than 100cm(1m)).\n"
    "3. Ask the question in this way: 'which is closer to the target object, object A or object B?'. A. Object A\nB. Object B\n"
    "4. Compare the distances of object A and object B to the target object, and give the correct answer"
    "[Important Attention]:\n"
    "1. Use object name to replace the object A, object B and target object in the question stem"
    "2. Do not give your thinking steps, just give the question and answer in the output format form I give you directly.\n"
    "3. Make sure that the question number is continuous.\n"
    "[Output Format]:\n"
    "Question 1: Which is closer to the target object, object A or object B?\n\n A. Object A\n B. Object B\n\n"
    "Answer 1: B. Object B\n\n"
)

prompt_bg_state = (
    "Your task is to ask **state questions about the objects** in the scene according to the ground truth information we provide to you."
    "This is a question about object's state, such as 'What's the state of object?'"
    "[Question Type]:\n"
    "The question type should be **objects'states question**, such as 'What's the state of the man'.\n"
    "Please make sure that every question has four options containing one definite and true answer\n"
    "[Ground Truth Information Format]:\n"
    "an example:\n"
    "Object: object name\n"
    "  State: crouch\n"
    "  Feature: wearing blue plaid pants\n"
    "[Think Step by Step]:\n"
    "You can do this step by step:\n"
    "1. Choose one object, and figure out it's state(**Make sure the state is not None!**)"
    "2. Ask question about object state question, such as 'What's the state of the object?'"
    "3. Given ground truth information, make sure the answer is correct and there exist four options to choose.\n"
    "[Attention]:\n"
    "1. Make full use of the ground truth state information provided.\n"
    "2. We will provide you an image caption containing objects in the scene, maybe you can use it to generate interesting questions.\n"
    "3. **Do not give away state information in your question stem** like 'What's the state of the man in blue plaid pants lying on the suburban street?'!That is meaningless."
    "[Output Format]:\n" 
    "Question 1: What's the state of the man?\n\n A. lying down\n B. standing\n C. crouch\n D. others\n\n"
    "Answer 1: A. lying down\n\n"
)

prompt_bg_cnt = (
    "Your task is to ask **object counting questions** in the scene according to the ground truth information we provide to you."
    "This is a question about counting objects in the scene, such as 'How many man in the scene?'."
    "[Question Type]:\n"
    "The question type should be **objects counting**, such as 'How many man in the scene'.\n"
    "This is an open-end question. That means,there is no need for you to design options.Just give question and answer directly\n"
    "[Ground Truth Information Format]:\n"
    "We simple summarize some ground truth which you can use directly.They are organized in this way:\n"
    " Total object: a number\n"
    " Total object type: a number\n"
    " Object number per each state:\n"
    "   crouch: a number\n"
    "   lying down: a number\n"
    "   standing: a number\n"
    " Object number per each object type and state:\n"
    "   player (crouch): a number\n"
    "   player (lying down):a number\n"
    "   player (standing): a number\n"
    "We also provide four features of human(player) to you.You can summarize the sharing features of them and ask interesting counting questions.\n"

    "[Think Step by Step]:\n"
    "You can do this step by step:\n"
    "Firstly, you can directly ask question using the summarized counting ground truth, such as 'How many people are in a crouch position in the scene?'"
    "Secondly, for the feature description, you can summarize the sharing features of objects (such as bald head, gender, skin color, wearning glasses or not...) and ask counting questions using these features, such as 'How many bald man in the scene?'"
    "[Some Tips]:\n"
    "1. Make full use of the ground truth information provided.\n"
    "2. You can summarize some sharing features of objects such as 'bald man', 'wether wearing glasses or not', and ask object counting question via these features.For example, 'How many bald mans in the scene?"
    "3. We will provide you an image caption containing objects in the scene, maybe you can use it to generate interesting questions.\n"
    "4. Counting numbers must be expressed in English, such as 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'.\n"
    "5. When you summarize the sharing features of objects, please make sure that the feature is common and not so specific\n"
    "6. If objects are not enough to ask questions, you can only use the ground truth information to generate questions\n"
    "[Output Format]:\n" 
    "Question 1: How many man in the scene?\n\n"
    "Answer 1: A. one\n\n"
)

prompt_bg_attribute =(
    "You should ask **object attribute questions about the objects** in the scene according to the ground truth information we provide to you."
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

name2typid = {
    "the_man_in_light_pants_and_sunglasses": "player_1",
    "the_man_in_green_sweater": "player_2",
    "the_woman_in_deep_V_top": "player_3",
    "the_woman_in_floral_shirt": "player_4",
    "the_woman_in_camo_vest": "player_5",
    "the_woman_in_black_vest": "player_6",
    "the_bald_man_in_glasses": "player_7",
    "the_bald_man_in_burgundy_sweaters_and_grey_suit": "player_8",
    "the_man_in_blue_polo": "player_9",
    "the_man_in_burgundy_polo_and_light_pants": "player_10",
    "the_man_in_white_shirt": "player_11",
    "the_man_in_grey_suit": "player_12",
    "the_man_in_blue_vest": "player_13",
    "the_bald_man_in_brown_sweater_and_black_pants": "player_15",
    "the_man_in_fedora_hat": "player_16",
    "the_man_in_blue_plaid_pants_and_light_brown_turtleneck_sweater": "player_17",
    "the_man_in_black_turleneck_sweater": "player_19",
    "Beagle_Dog": "animal_0",
    "Great_Dane": "animal_1",
    "Doberman_Pinscher": "animal_2",
    "Tabby_Cat": "animal_3",
    "Water_Buffalo": "animal_6",
    "Komodo_Dragon": "animal_10",
    "Pig": "animal_11",
    "Spider": "animal_12",
    "Camel": "animal_14",
    "Horse": "animal_15",
    "Puma": "animal_16",
    "Penguin": "animal_19",
    "Rhinoceros": "animal_20",
    "Tiger": "animal_21",
    "Zebra": "animal_22",
    "Elephant": "animal_23",
    "Turtle": "animal_25",
    "Snapping_Turtles": "animal_26",
    "Toucan": "animal_27",
    "White_Drone": "drone_0",
    "Grey_SUV": "car_0",
    "White_Compact_Car": "car_1",
    "Orange_Sedan_Car": "car_2",
    "Yellow_Car": "car_3",
    "Red_Motorbike": "motorbike_0"
} 

class generate_qa_agent:
    def __init__(self, model_name="doubao",base_dir= None,model_config_path="model_config.json"):
        self.base_dir = base_dir
        self.model = model_name
        initialize_model(model_name,model_config_path)


    def format_dict_for_llm(self, gt_information, question_type = "relative_location",agent_cnt_feature_path = None):

        import numpy as np
        
        result = "\n"
        
        def format_array(arr):
            if isinstance(arr, np.ndarray):
                arr = arr.tolist()
            return str(arr)
        
        result = "\n"
        for obj_name, basic_info in gt_information["agent_info"].items():
            result += f"Object: {obj_name}\n"
            result += f"  State: {basic_info['state']}\n"
            result += f"  Feature: {basic_info['feature']}\n\n"
        
        if question_type == "state":
            return result
        elif question_type == "counting":
            with open(agent_cnt_feature_path, 'r') as f:
                agent_cnt_features = json.load(f)
            
            cnt_gt = "[counting ground truth]:\n"
            cnt_info = gt_information["counting_gt"]
            cnt_gt += f"  Total objects: {cnt_info['total_agents']}\n"
            cnt_gt += f"  Total object type: {cnt_info['total_agent_types']}\n"
            cnt_gt += f"  Object number per each state:\n"
            for state, count in cnt_info["agents_per_state"].items():
                cnt_gt += f"    {state}: {count}\n"
            cnt_gt += f"  Object number per each object type and state:\n"
            for obj_type_state, state_counts in cnt_info["agents_per_type_state"].items():
                splits = obj_type_state.split('_')
                type = splits[0]
                state = "_".join(splits[1:])
                cnt_gt += f"    {type} ({state}): {state_counts}\n"
            feature_description = "[feature description]:\n"
            for agent_name in gt_information["agent_info"].keys():
                typid = name2typid[agent_name]
                type, id = typid.split('_') 
                if type == "player":
                    feature_dict = agent_cnt_features[type][id]
                    feature_description += f"  {agent_name}, gender:{feature_dict['Gender']}, skin color:{feature_dict['Skin_color']}, bald or not:{feature_dict['Bald_or_not']},wearing glasses or not:{feature_dict['Wearing_glasses_or_not']}\n"
            feature_description += "\n"
            return cnt_gt + feature_description
        
        elif question_type == "relative_location":
            rel_loca = "\n"
            rel_loca_info = gt_information["relative_location"]
            for obj_name, obj_info in gt_information["agent_info"].items():
                rel_loca += f"Object: {obj_name}, State: {obj_info['state']}\n"
                rel_obj_info = rel_loca_info[obj_name]
                for obj_name2, rel_info in rel_obj_info.items():
                    relative_location = rel_info['relative_direction']
                    rel_loca += f"  The {obj_name2} is {relative_location} the {obj_name}\n"
            return rel_loca
        elif question_type == "relative_distance":
            rel_distance = "\n"
            rel_distance_info = gt_information["relative_location"]
            for obj_name, basic_info in gt_information["agent_info"].items():
                rel_distance += f"Object: {obj_name}, State: {basic_info['state']}\n"
                rel_obj_info = rel_distance_info[obj_name]
                for obj_name2, rel_info in rel_obj_info.items():
                    relative_distance = rel_info['distance']
                    rel_distance += f"  The distance between {obj_name2} and {obj_name} is {relative_distance}\n"

            return rel_distance

    def encode_image_array(self, image_array):
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

    def image_captioning(self, image_list): # Modified to accept a list of images


        user_message_content = []
        # Add a text part first, if desired, to guide the model
        # user_message_content.append({"type": "text", "text": "Please describe the scene depicted in the following images:"})

        for image_array in image_list:
            base64_image = self.encode_image_array(image_array)
            user_message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"} # Assuming PNG
                }
            )
        
        if not user_message_content:
            print("Warning: No images provided for captioning.")
            return "No images were provided to describe."

        response = client.chat.completions.create(
            model=self.model,
            max_tokens=1000, # Adjust as needed
            messages=[
                {"role": "system", "content": scene_prompt}, # System prompt
                {"role": "user", "content": user_message_content} # User content now a list of image URLs
            ],
        )
        respon = response.choices[0].message.content
        print("Generated Caption:\n", respon)
        return respon

    def generate_question(self, gt_info, image_description, model, question_type = "relative_location"):    

        prompt = ""
        if question_type == "relative_location":
            prompt = prompt_bg_rela_loca
        elif question_type == "counting":
            prompt = prompt_bg_cnt
        elif question_type == "state":
            prompt = prompt_bg_state
        elif question_type == "relative_distance":
            prompt = prompt_bg_rela_distance
        else:
            raise ValueError("Invalid question type. Please choose 'relative_location', 'relative_rotation', 'counting', or 'state'.")
            
        response = client.chat.completions.create(
            model=model,
            max_tokens=10000,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": f"scenary description:{image_description}"
                    },
                    {
                        "type":"text",
                        "text": f"ground truth information:{gt_info},it contains the ground truth information of the objects in the scene"
                    },
                    {
                        "type": "text",
                        "text": f"Please ask **object {question_type}** questions about the objects in the scene according to the ground truth information and scene description."
                    }
                    ]
                },
            ],

        )
        respon = response.choices[0].message.content
        return respon

    def save_collection_data_to_single_file(self, qa_string, image_list,target_configs, refer_agents_category, safe_start,
                                            env_name="SuburbNeighborhood_Day", question_type="relative_location", 
                                            batch_id=None, sample_configs=None):
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
        save_dir = os.path.join(self.base_dir, env_name)
        save_dir = os.path.join(save_dir, question_type, batch_id)
        os.makedirs(save_dir, exist_ok=True)
        
        # Define file path (single file for all batches)
        data_file = os.path.join(save_dir, f"{question_type}.json")
        
        # Save observations
        obs_folder = f"obs_{batch_id}"
        obs_dir = os.path.join(save_dir, obs_folder)
        os.makedirs(obs_dir, exist_ok=True)
        for i, obs in enumerate(image_list):
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
            cv2.imwrite(os.path.join(obs_dir, f"obs_{i}.png"), obs)

        questions_dict = {}
        if question_type != 'counting':
            pattern = r'Question (\d+): (.*?)\n\n((?:\s*[A-D]\.\s.*?\n)+)\n*Answer \1: ([A-D]\..*?)(?:\n\n|$)'
        else:
            # Regex for counting questions (open-ended answer, possibly with a single letter option like "A.")
            pattern = r'Question (\d+): (.*?)\n\nAnswer \1: (.*?)(?:\n\n|$)'
        
        matches = re.finditer(pattern, qa_string, re.DOTALL)
        
        for match in matches:
            q_num = match.group(1)
            question_text = match.group(2).strip()
            
            options = []  # Initialize options as an empty list
            
            if question_type != 'counting':
                # For non-counting questions, options are in group 3, answer in group 4
                options_text = match.group(3).strip()
                answer_text = match.group(4).strip()
                for option_line in options_text.split('\n'):
                    if option_line.strip():
                        options.append(option_line.strip())
                questions_dict[f"Question {q_num}"] = {
                "question": question_text,
                "options": options, # For counting, this will be an empty list
                "answer": answer_text
                }
            else:
                answer_text = match.group(3).strip()        
                questions_dict[f"Question {q_num}"] = {
                    "question": question_text,
                    "answer": answer_text
                }
        # Create batch data
        filtered_target_configs = {}
        for category in refer_agents_category:
            if category in target_configs:
                filtered_target_configs[category] = target_configs[category]

        batch_data = {
            "target_configs": filtered_target_configs, 
            "sample_configs": sample_configs,  # Add sample_configs if available
            "refer_agents_category": refer_agents_category,
            "safe_start": safe_start,
            "obs_dir": obs_dir,
            "Questions": questions_dict
        }
        file_data = batch_data
        with open(data_file, 'w') as f:
            json.dump(file_data, f, indent=4)
        print(f"Batch {batch_id} saved to {data_file}")
        return batch_id

def check_reach(goal, now):
    error = np.array(now[:2]) - np.array(goal[:2])
    distance = np.linalg.norm(error)
    return distance < 40

if __name__ == '__main__':
    env_name = "Map_ChemicalPlant_1" # Change this to the desired environment name
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env_id", nargs='?', default=f'UnrealCvEQA_DATA-{env_name}-DiscreteColorMask-v0',
                        help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=0, help='random seed')
    parser.add_argument("-t", '--time-dilation', dest='time_dilation', default=-1, help='time_dilation to keep fps in simulator')
    parser.add_argument("-n", '--nav-agent', dest='nav_agent', action='store_true', help='use nav agent to control the agents')
    parser.add_argument("-d", '--early-done', dest='early_done', default=-1, help='early_done when lost in n steps')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')
    parser.add_argument("-c", '--ifonly-counting', dest='if_cnt', action='store_true', help='if only counting the number of agents in the scene')
    parser.add_argument("-g", "--reachable-points-graph", dest='graph_path', default=f"./agent_configs_sampler/points_graph/{env_name}/environment_graph_1.gpickle", help='use reachable points graph')
    parser.add_argument("-w", "--work-dir", dest='work_dir', default="E:/EQA/unrealzoo_gym/example", help='work directory to save the data')
    parser.add_argument("--config-path", dest='config_path', default=os.path.join(current_dir, "solution"), help='path to model config file')
    parser.add_argument("--model", dest="model", default="gemini_pro", help="model name")
    # parser.add_argument('--config', type=str, default='model_config.json', help='Path to model configuration file (default: model_config.json)')
    args = parser.parse_args()
    env_list = [
    # "Map_ChemicalPlant_1",
    # "ModularNeighborhood",
    # "ModularSciFiVillage",
    # "RuralAustralia_Example_01",
    # "ModularVictorianCity",
    # "Cabin_Lake",
    # "Pyramid"
    # "ModularGothic_Day",
    # "Greek_Island"
    "SuburbNeighborhood_Day"
    ]
    curr_path = os.path.dirname(os.path.abspath(__file__))
    gt_path = f"{curr_path}/GT_info"
    model_config_path = f"{curr_path}/solution/model_config.json"
    gen_qa_agent = generate_qa_agent(model_name=args.model,model_config_path=model_config_path)
    agent_cnt_feature_path = f"{curr_path}/agent_caption/agent_cnt_features.json"
    type_list = ["relative_location"]
    for env_name in env_list:
        try:
            with open(os.path.join(gt_path, env_name, "status_recorder.json"), 'r') as f:
                status_recorder = json.load(f)
            processed_any = False
            for fodername, status in status_recorder.items():
                # if status:
                #     print(f"Skipping {fodername} as it has already been processed.")
                #     continue
                # else:
                    print(f"Processing {fodername}...")
                    # status_recorder[filename] = True
                    gt_info_path = os.path.join(gt_path, env_name, fodername, "gt_info.json")
                    obs_path = os.path.join(gt_path, env_name, fodername, "obs")
                    with open(gt_info_path, 'r') as f:
                        gt_info = json.load(f)
                    visit_id = fodername
                    # record information to store
                    target_configs = gt_info["target_configs"]
                    sample_configs = gt_info["sample_configs"]
                    gt_information = gt_info["gt_information"]
                    obs_names = gt_info["obs_filenames"]
                    sample_configs = gt_info["sample_configs"]
                    refer_agents_category = list(gt_info["target_configs"].keys())
                    safe_start = gt_info["safe_start"]
                    obs_rgb = []
                    # load obs
                    for obs_name in obs_names:
                        path = os.path.join(obs_path, obs_name)
                        if os.path.exists(path):
                            img = cv2.imread(path)
                            if img is not None:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                obs_rgb.append(img)
                            else:
                                print(f"Warning: {path} is not a valid image file.")
                        else:
                            print(f"Warning: {path} does not exist.")
                    image_caption = gen_qa_agent.image_captioning(obs_rgb) # Generate image captioning
                    # generate question type by type
                    for question_type in type_list:
                        print(f"Generating questions for {question_type}...")          
                        gt_info_str = gen_qa_agent.format_dict_for_llm(gt_information, question_type,agent_cnt_feature_path = agent_cnt_feature_path)
                        respon = gen_qa_agent.generate_question(gt_info_str, image_caption, question_type) 
                        print(f"\nGenerated questions for {question_type}:\n", respon)  
                        gen_qa_agent.save_collection_data_to_single_file(respon, obs_rgb, target_configs, refer_agents_category, safe_start,env_name=env_name,question_type=question_type,batch_id = visit_id, sample_configs = sample_configs)
                    # Update status recorder
                    processed_any = True
                    status_recorder[fodername] = True
            if processed_any:       
                with open(os.path.join(gt_path, env_name, "status_recorder.json"), 'w') as f_write:
                    json.dump(status_recorder, f_write, indent=4)
                print("All processed status updated.")
                    
        except FileNotFoundError:
            print(f"Error: status_recorder.json or other essential file not found. Please check paths.")
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from a file. Ensure JSON files are correctly formatted.")
        except Exception as e: 
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc() 

        finally: 
            # f.close() if 'f' in locals() and not f.closed else None
            print("Processing complete. All files closed.")