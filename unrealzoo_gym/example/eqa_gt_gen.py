import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
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
    elif key_state['n']:
        collection = 2
    elif key_state['c']:
        collection = 3
    return collection

import json
import os


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

if __name__ == '__main__':
    # env name
    env_list = [
    # "Map_ChemicalPlant_1",
    # "ModularNeighborhood",
    # "ModularSciFiVillage",
    # "RuralAustralia_Example_01",
    # "ModularVictorianCity",
    # "Cabin_Lake",
    # "Pyramid",
    "ModularGothic_Day",
    "Greek_Island"
    ]
    env_name = "Greek_Island"  # Change this to the desired environment name
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
    parser.add_argument("-g", "--reachable-points-graph", dest='graph_path', default=f"./agent_configs_sampler/points_graph/{env_name}/environment_graph.gpickle", help='use reachable points graph')
    parser.add_argument("-w", "--work-dir", dest='work_dir', default="E:/EQA/unrealzoo_gym/example", help='work directory to save the data')
    args = parser.parse_args()
    
    env = gym.make(args.env_id)
    
    # some configs
    currpath = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(currpath, args.graph_path)
    agents_category = ['player','drone', 'animal']
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
    
    # agent sample pre-define
    agents_categories = ['player','drone', 'animal','car']     
    type_ranges = {
        'player': (4, 6),
        'car': (1, 2),     
        'animal': (0, 1),   
        'drone': (0, 1),
        'motorbike': (0, 0)
    }
    min_total = 5
    max_total = 8

    # wrapper
    # env = augmentation.RandomPopulationWrapper
    env = sample_agent_configs.SampleAgentConfigWrapper(
        env,
        agent_category=agents_categories,
        min_types=1,  
        max_types=4,
        type_count_ranges=type_ranges,       
        min_total_agents=min_total,  
        max_total_agents=max_total,
        graph_path=graph_path,
        if_cnt=args.if_cnt 
    )

    env = configUE.ConfigUEWrapper(env, offscreen=False)
    save_cnt = 0
    try:
        while save_cnt <= 5:
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
                base_gt_path = os.path.join(args.work_dir, f"GT_info/{env_name}")
            else:
                base_gt_path = os.path.join(args.work_dir, f"GT_info/{env_name}_counting_only")
            current_gt_info = os.path.join(base_gt_path, str(instance_id)) # including gt_info.json and obs
            current_instance_obs_path = os.path.join(current_gt_info , f"obs")
            gt_info_filename = os.path.join(current_gt_info, f"gt_info.json")
            record_file = os.path.join(base_gt_path, f"status_recorder.json")

            # sample obs
            collected_images_for_instance = []  # record obs
            cam_position = env.unwrapped.camera_position
            # start keyboard listener   
            listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            listener.start()
            print(f"\nProcessing Batch: {batch}, Instance ID: {instance_id}")
            safe_start_flag = 0
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
                # bgr_image_from_env = env.unwrapped.unrealcv.read_image(1,"lit","direct")
                # mask_obs = env.unwrapped.unrealcv.read_image(1,"object_mask","direct")

                if bgr_image_from_env.dtype != np.uint8:
                    if bgr_image_from_env.max() <= 1.0 and bgr_image_from_env.min() >= 0.0:
                        bgr_image_from_env = (bgr_image_from_env * 255).astype(np.uint8)
                    else:
                        bgr_image_from_env = bgr_image_from_env.astype(np.uint8)
                
                cv2.imshow("RGB/BGR", bgr_image_from_env) 
                cv2.imshow("Mask", mask_obs)
                cv2.waitKey(0)

                print(f"  Camera {cam_idx + 1}/{len(cam_position)}: Press 'y' to save, 'n' to skip.")
                collection_choice = 0
                key_state['y'] = False 
                key_state['n'] = False
                key_state['c'] = False
                flag = 0
                while collection_choice == 0:
                    collection_choice = get_key_collection()
                    time.sleep(0.05)
                
                if collection_choice == 1: # 'y'
                    collected_images_for_instance.append((bgr_image_from_env, cam_idx))
                    if safe_start_flag == 0:
                        # set safe start
                        safe_start = position
                        safe_start_flag = 1
                    print(f"    Image from camera {cam_idx + 1} marked for saving.")
                elif collection_choice == 2: # 'n'
                    print(f"    Image from camera {cam_idx + 1} skipped.")
                elif collection_choice == 3: # 'c'
                    print("    Collection cancelled.")
                    flag = 1
                    break
                time.sleep(0.1)
            listener.stop() 
            if args.render:
                cv2.destroyAllWindows()

            if flag == 1:
                print("Collection cancelled.")
                continue
            
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
                "safe_start": safe_start,
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
    except KeyboardInterrupt:
        print('\nExiting due to KeyboardInterrupt...')
        if 'listener' in locals() and listener.is_alive():
            listener.stop()
        env.close()
    finally:
        if args.render:
            cv2.destroyAllWindows()




