import os
import numpy as np
import json
import unrealcv

def load_env_setting(filename):
    f = open(get_settingpath(filename))
    type = os.path.splitext(filename)[1]
    assert type in ['.json'], type + ' is not supported'
    setting = json.load(f)
    return setting


def get_settingpath(filename):
    import gym_unrealcv
    gympath = os.path.dirname(gym_unrealcv.__file__)
    return os.path.join(gympath, 'envs', 'setting', filename)

def get_action_size(action):
    return len(action)

def get_direction(current_pose, target_pose):  # get relative angle between current pose and target pose in x-y plane
    y_delt = target_pose[1] - current_pose[1]
    x_delt = target_pose[0] - current_pose[0]
    if x_delt == 0 and y_delt == 0:  # if the same position
        return 0
    angle_abs = np.arctan2(y_delt, x_delt)/np.pi*180
    angle_relative = angle_abs - current_pose[4]
    if angle_relative > 180:
        angle_relative -= 360
    if angle_relative < -180:
        angle_relative += 360
    return angle_relative


def get_textures(texture_name="textures", docker=False):
    try:
        texture_dir = os.path.join(unrealcv.util.get_path2UnrealEnv(), "textures")
    except AttributeError:
        raise ImportError(
            "Function get_path2UnrealEnv() not found. "
            "Please upgrade unrealcv to version 1.1.5 or higher using: \n"
            "pip install --upgrade unrealcv"
            )
    textures_list = os.listdir(texture_dir)
    # relative to abs
    for i in range(len(textures_list)):
        if docker:
            textures_list[i] = os.path.join('/unreal', texture_dir, textures_list[i])
        else:
            textures_list[i] = os.path.join(texture_dir, textures_list[i])
    return textures_list

def convert_dict(old_dict):
    new_dict = {}
    for agent, info in old_dict.items():
        names = info["name"]
        for i, name in enumerate(names):
            new_dict[name] = {
                "agent_type": agent,
            }
            for key in info.keys():
                if key == "name" or key == "cam_id" or key == "class_name" or key == "start_pos" or key == "animation" or key == "feature_caption" or key == "app_id" or key == "type":
                    new_dict[name][key] = info[key][i]
                else:
                    new_dict[name][key] = info[key]
    return new_dict