import gym
from gym import Wrapper
import json
def load_json_file(file_path):
    """
    Load a JSON file and return its content.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        dict: Content of the JSON file.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

class ConfigUEWrapper(Wrapper):
    def __init__(self, env, docker=False, resolution=(160, 160), display=None, offscreen=False,
                            use_opengl=False, nullrhi=False, gpu_id=None, sleep_time=5, comm_mode='tcp',name_mapping_dict_path=None):
        super().__init__(env)
        env.unwrapped.docker = docker
        env.unwrapped.display = display
        env.unwrapped.offscreen_rendering = offscreen
        env.unwrapped.use_opengl = use_opengl
        env.unwrapped.nullrhi = nullrhi
        env.unwrapped.gpu_id = gpu_id
        env.unwrapped.sleep_time = sleep_time
        env.unwrapped.resolution = resolution
        env.unwrapped.comm_mode = comm_mode
        if name_mapping_dict_path is not None:
            env.unwrapped.name_mapping_dict = load_json_file(name_mapping_dict_path)
        else:
            env.unwrapped.name_mapping_dict = None

    def step(self, action):
        obs,reward,termination,truncation,info = self.env.step(action)
        return obs,reward,termination,truncation,info

    def reset(self, **kwargs):
        states,info = self.env.reset(**kwargs)
        return states, info