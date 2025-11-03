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

# 全局客户端和模型配置
client = None
current_model_name = None
current_model_config = None

def initialize_model(model_name="doubao",model_config_path="model_config.json"):
    """初始化指定的模型"""
    global client, current_model_name, current_model_config
    client, current_model_name, current_model_config = create_client(model_name,model_config_path)
    print(f"已初始化模型: {model_name} ({current_model_name})")


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

def call_api_vlm(sys_prompt, usr_prompt, base64_image_list=[]):
    """
    Call the vLM API with the given prompt.
    """
    messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {
                    "type": "text",
                    "text": usr_prompt,
                }
            ]
            }
    ]

    for base64_image in base64_image_list:
            messages.append({"role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    }
                }]
            })
    # Assuming OpenAI API is set up correctly
    response = client.chat.completions.create(
        model=current_model_name,
        max_tokens=10000,
        messages=messages
    )
    respon=  response.choices[0].message.content.strip()
    print(f"[VLM RESPONSE] {respon}")
    return respon

def encode_image_array(image_array):
    from PIL import Image
    import io
    import base64
    # Convert the image array to a PIL Image object
    image = Image.fromarray(np.uint8(image_array))

    # Save the PIL Image object to a bytes buffer
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")

    # Encode the bytes buffer to Base64
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return img_str
    
sys_prompt = """
    You are a smart environment examiner.
    Your task is to examine whether the environment is naturally and reasonably populated with agents.
    You will be provided with a first-person view image.
    You need to check whether the vehicle is damaged and whether the people are trapped.
    output format:
    use xml format to output your answer.
    <a>yes/no</a>
"""

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


if __name__ == '__main__':
    # env name
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env_name", nargs='?', default="Map_ChemicalPlant_1", help='Select the environment name to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=0, help='random seed')
    parser.add_argument("-t", '--time-dilation', dest='time_dilation', default=-1, help='time_dilation to keep fps in simulator')
    parser.add_argument("-n", '--nav-agent', dest='nav_agent', action='store_true', help='use nav agent to control the agents')
    parser.add_argument("-d", '--early-done', dest='early_done', default=-1, help='early_done when lost in n steps')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')
    parser.add_argument("-c", '--ifonly-counting', dest='if_cnt', action='store_true', help='if only counting the number of agents in the scene')
    # parser.add_argument("-g", "--reachable-points-graph", dest='graph_path', default=f"./agent_configs_sampler/points_graph/{env_name}/environment_graph_1.gpickle", help='use reachable points graph')
    # parser.add_argument("-w", "--work-dir", dest='work_dir', default=current_dir, help='work directory to save the data')
    parser.add_argument("--config-path", dest='config_path', default=os.path.join(current_dir, "solution"), help='path to model config file')
    parser.add_argument("--model", dest="model", default="gemini_pro", help="model name")
    # parser.add_argument("--floor_height", dest="floor_height", type=float, default=-12776.0, help="floor height from the ground")
    parser.add_argument("--camera_height", dest="camera_height", type=int, default=1200, help="camera height from the ground")
    parser.add_argument("-p", "--pid_port", type=int, default=50007, help="UnrealCV watchdog pid")
    parser.add_argument("--use_pid", type=bool,default=False, help="Whether to use pid watchdog to monitor the UE process")
    parser.add_argument("--ue_log_dir", default=os.path.join(current_dir, "unreal_log_path"), help="unreal engine logging directory")
    # parser.add_argument("--agent_categories", type=str, help="JSON encoded agent categories")
    # parser.add_argument("--type_ranges", type=str, help="JSON encoded type ranges")
    # parser.add_argument("--min_total", type=int, help="Minimum total number of agents")
    # parser.add_argument("--max_total", type=int, help="Maximum total number of agents")
    parser.add_argument('--status_path', default=f"{current_dir}/GT_info",help="场景路径")
    args = parser.parse_args()

    env_dict = {
        # "Map_ChemicalPlant_1":{ 
        #     "height": -12776.0,
        #     "agent_categories": ['player','drone','car','animal'],
        #     },
        # "SuburbNeighborhood_Day":{
        #     "height": 0,
        #     "agent_categories": ['player','drone','car']
        # },
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
        #     "agent_categories": ['player','drone','animal']
        # },
        # "Demonstration_Castle": {
        #     "agent_categories": ['player','drone','animal']
        # },
        "Venice": {
            "agent_categories": ['player','drone','animal']
        },
    }

    all_type_ranges = {
        'player': (2, 6),
        'car': (0, 2),
        'drone': (0, 1),
        'animal': (0, 2),
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
            env_id = f'UnrealCvEQA_DATA-{env_name}-DiscreteColorMask-v0'
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
                    "BP_tree31_47",
                    "BP_tree32_50",
                    "BP_tree33_53",
                    "BP_tree34_59",
                    "BP_tree35_62",
                    "BP_tree36_71",
                    "BP_tree37_74",
                    "BP_tree38_2",
                    "BP_tree39_5",
                    "BP_tree3_14",
                    "BP_tree40_8",
                    "BP_tree41_11",
                    "BP_tree42_14",
                    "BP_tree43_17",
                    "BP_tree44_20",
                    "BP_tree45_32",
                    "BP_tree46_35",
                    "BP_tree47_38",
                    "BP_tree48_41",
                    "BP_tree49_44",
                    "BP_tree4_25",
                    "BP_tree50_47",
                    "BP_tree51_50",
                    "BP_tree54_29",
                    "BP_tree5_28",
                    "BP_tree6_31",
                    "BP_tree7_37",
                    "BP_tree8_40",
                    "BP_tree9_46",
                    "BP_tree_20"
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
                obj_2_hide=obj_2_hide 
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
                while save_cnt <= 11:
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
                        base_gt_path = os.path.join(current_dir, f"GT_info/{env_name}")
                    else:
                        base_gt_path = os.path.join(current_dir, f"GT_info/{env_name}_counting_only")
                    current_gt_info = os.path.join(base_gt_path, str(instance_id)) # including gt_info.json and obs
                    current_instance_obs_path = os.path.join(current_gt_info , f"obs")
                    gt_info_filename = os.path.join(current_gt_info, f"gt_info.json")
                    record_file = os.path.join(base_gt_path, f"status_recorder.json")
                    trace_dir = os.path.join(current_gt_info, f"trace")
                    # sample obs
                    collected_images_for_instance = []  # record obs
                    cam_position = env.unwrapped.camera_position
                    # start keyboard listener   
                    # listener = keyboard.Listener(on_press=on_press, on_release=on_release)
                    # listener.start()
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

                        # update_camera_configs
                        env.unwrapped.unrealcv.cam = env.unwrapped.unrealcv.get_camera_config()
                        env.unwrapped.update_camera_assignments()
                        cam_id = env.unwrapped.agents[obs_name]['cam_id']

                        if bgr_image_from_env.dtype != np.uint8:
                            if bgr_image_from_env.max() <= 1.0 and bgr_image_from_env.min() >= 0.0:
                                bgr_image_from_env = (bgr_image_from_env * 255).astype(np.uint8)
                            else:
                                bgr_image_from_env = bgr_image_from_env.astype(np.uint8)
                        # set safe start
                        safe_start_to_collect.append(position)
                        print(f"    Image from camera {cam_idx + 1} marked for saving.")
                        collected_images_for_instance.append((bgr_image_from_env, cam_idx))
                        # collect trajectory
                        # type_2_sample = list(current_target_configs.keys())
                        # if 'drone' in type_2_sample:
                        #     type_2_sample.remove('drone')
                        # agent_type = random.choice(type_2_sample)
                        # agent_name = random.choice(current_target_configs[agent_type]['name'])
                        # name_index = current_target_configs[agent_type]['name'].index(agent_name)
                        # loc = current_target_configs[agent_type]['start_pos'][name_index]
                        # cur_location = position[:3] 
                        # # loc_2_sample = current_target_configs[agent_type]['start_pos']
                        # # loc = random.choice(loc_2_sample)

                        # img_list = []
                        # pose_list = []
                        # action_list = []
                        # time_list = []
                        # count_step = 0
                        # path_string =env.unwrapped.unrealcv.nav_to_goal_bypath(obs_name,loc[:3])
                        # print("=========path_string=========\n",path_string)
                        # while check_reach(loc, cur_location):
                        #     obs=env.unwrapped.unrealcv.read_image(cam_id, 'lit', mode='direct')
                        #     cur_pose = env.unwrapped.unrealcv.get_obj_pose(obs_name)
                        #     time_list.append(time.time())
                        #     cur_location = cur_pose[:3]
                        #     img_list.append(obs)
                        #     pose_list.append(cur_pose)
                        #     count_step += 1
                        # # caculate action for each frame
                        # for i in range(0, len(pose_list)-1):
                        #     delta_yaw = pose_list[i+1][-2] - pose_list[i][-2]
                        #     direct_vec = np.array(pose_list[i+1][:2]) - np.array(pose_list[i][:2])
                        #     delta_t = time_list[i+1] - time_list[i]
                        #     distance = np.linalg.norm(direct_vec)
                        #     linear_velocity = distance / delta_t
                        #     angle_velocity = delta_yaw / delta_t
                        #     action_list.append([angle_velocity, linear_velocity])
                        # data_to_save = {}
                        # data_to_save["Instruction"] = f"{info['Instruction']} {info['Relative_pose']} you"
                        # data_to_save["Target_Type"] = agent_type
                        # data_to_save["Start_Pose"] = position
                        # data_to_save["Target_Name"] = agent_name
                        # data_to_save["Target_Pose"] = loc
                        # data_to_save["Trajectory_Pose"] = pose_list
                        # data_to_save["Action_Per_Frame"] = action_list
                        # data_to_save["Time_Per_Frame"] = time_list
                        # save_data(data_to_save, img_list,f"{trace_dir}/{trace_cnt}")
                        # trace_cnt+=1
                    # if args.render:
                    # cv2.destroyAllWindows()
                    
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
            finally:
                env.close()
                exit(-1)




