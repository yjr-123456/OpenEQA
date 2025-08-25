import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import argparse
#import gym_unrealcv
import gym
from gym import wrappers
import cv2
import time
import numpy as np
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, agents, augmentation, configUE, sample_agent_configs
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
            api_key="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"
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


def check_reach(goal, now):
    error = np.array(now[:2]) - np.array(goal[:2])
    distance = np.linalg.norm(error)
    return distance < 40


if __name__ == '__main__':
    # env name
    env_name = "SuburbNeighborhood_Day"
    parser = argparse.ArgumentParser(description=None)
    # parser.add_argument("-e", "--env_id", nargs='?', default='UnrealTrack-track_train-ContinuousMask-v4',
    #                     help='Select the environment to run')
    parser.add_argument("-e", "--env_id", nargs='?', default=f'UnrealCvEQA_DATA-{env_name}-DiscreteRgbd-v0',
                        help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=0, help='random seed')
    parser.add_argument("-t", '--time-dilation', dest='time_dilation', default=-1, help='time_dilation to keep fps in simulator')
    parser.add_argument("-n", '--nav-agent', dest='nav_agent', action='store_true', help='use nav agent to control the agents')
    parser.add_argument("-d", '--early-done', dest='early_done', default=-1, help='early_done when lost in n steps')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')
    parser.add_argument("-g", "--reachable-points-graph", dest='graph_path', default=f"./agent_configs_sampler/points_graph/{env_name}/environment_graph.gpickle", help='use reachable points graph')
    args = parser.parse_args()
    
    env = gym.make(args.env_id)
    
    # some configs
    currpath = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(currpath, args.graph_path)
    agents_category = ['player','drone','motorbike','animal']
    


    # env wrappers
    if int(args.time_dilation) > 0:  # -1 means no time_dilation
        env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
    if int(args.early_done) > 0:  # -1 means no early_done
        env = early_done.EarlyDoneWrapper(env, int(args.early_done))
    if args.monitor:
        env = monitor.DisplayWrapper(env)
    # env = augmentation.RandomPopulationWrapper(env, agent_num+1,agent_num+1, random_target=False)
    
    # wrappers
    env = sample_agent_configs.SampleAgentConfigWrapper(env, agents_category, min_types=3, max_types=4, graph_path=graph_path)
    env = configUE.ConfigUEWrapper(env, offscreen=False)

    # keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    try:
        for i in range(10):
            image = []   # record obs
            state, info = env.reset()
            obj_dict = info['object_dict']
            if obj_dict == {}:
                continue
            print("ground truth info: \n", info["gt_information"])
            gt_info_filename = f"gt_information_{i}.json"  # 使用循环变量 i 确保文件名唯一
            try:
                with open(gt_info_filename, 'w') as f:
                    json.dump(info["gt_information"], f, indent=4)
                print(f"Ground truth information saved to {gt_info_filename}")
            except Exception as e:
                print(f"Error saving ground truth information: {e}")
            obs_color = state[...,:3].squeeze()
            # cv2.imshow("obs", obs_color[0])

            # get some config information
            # target_configs = env.unwrapped.target_configs
            # agents_category = env.unwrapped.refer_agents_category
            # cam_position = env.unwrapped.camera_position
            # safe_start = env.unwrapped.safe_start
            # agent_num = env.unwrapped.num_agents

            # goal_idx = 0
            # cnt_step = 0
            # obj = env.unwrapped.player_list[0]
            # obj_cam_id = env.unwrapped.cam_list[0]
            # while True:
            #     action = [6]
            #     print("please get action!")
            #     while action == [6] and not any(key_state[k] for k in ['y', 'm', 'b']):
            #         action = get_key_action()
            #         time.sleep(0.1)
                
            #     if action != [6]:
            #         obs, reward, termination, truncation, info = env.step(action)
            #         obs_color = obs[...,:3].squeeze()
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
            #                 questions, 
            #                 image,
            #                 target_configs, 
            #                 agents_category, 
            #                 safe_start,
            #                 env_name=env_name,
            #                 question_type=question_type
            #             )
            #     print(f"Questions have been saved to batch {batch_id}")
        env.close()
    except KeyboardInterrupt:
        print('exiting')
        env.close()




