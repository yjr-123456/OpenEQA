from gymnasium import Wrapper
import numpy as np
from gym_unrealcv.envs.utils import misc
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json

class ConfigGenerator:
    @staticmethod
    def create_player_config(name,class_name,cam_id) -> dict:
        return dict(
            name=[name],
            cam_id=[cam_id],
            class_name=[class_name],
            internal_nav= True,
            scale=[1, 1, 1],
            relative_location=[20, 0, 0],
            relative_rotation=[0, 0, 0],
            head_action_continuous={
                "high": [15, 15, 15],
                "low":  [-15, -15, -15]
            },
            head_action=[
                [0, 0, 0], [0, 30, 0], [0, -30, 0]
            ],
            animation_action=["stand", "jump", "crouch"],
            move_action=[
                [0, 100], [0, -100], [15, 50], [-15, 50], [30, 0], [-30, 0], [0, 0]
            ],
            move_action_continuous={
                "high": [30, 100],
                "low": [-30, -100]
        }
        )
    @staticmethod
    def create_drone_config(name, class_name, cam_id) -> dict:
        return dict(
            name=[name],
            cam_id=[cam_id],
            class_name=[class_name],
            internal_nav=True,
            scale=[0.3, 0.3, 0.3],
            relative_location=[20, 0, 0],
            relative_rotation=[0, 0, 0],
            move_action=[
                [0.5, 0, 0, 0], [-0.5, 0, 0, 0],
                [0, 0.5, 0, 0], [0, -0.5, 0, 0],
                [0, 0, 0.5, 0], [0, 0, -0.5, 0],
                [0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 0]
            ],
            move_action_continuous={
                "high": [1, 1, 1, 1],
                "low": [-1, -1, -1, -1]
            }
        )
    @staticmethod
    def create_animal_config(name, class_name, cam_id) -> dict:
        return dict(
            name=[name],
            cam_id=[cam_id],
            class_name=[class_name],
            internal_nav=True,
            scale=[1, 1, 1],
            relative_location=[20, 0, 0],
            relative_rotation=[0, 0, 0],
            move_action=[
                [0, 200],
            [0, -200],
            [15, 100],
            [-15, 100],
            [30, 0],
            [-30, 0],
            [0, 0]
        ],
        move_action_continuous={
            "high": [30, 200],
            "low": [-30, -200]
        }
        )
    @staticmethod
    def generate_car_config(name, class_name, cam_id) -> dict:
        return dict(
            name=[name],
            cam_id=[cam_id],
            class_name=[class_name],
            internal_nav=True,
            scale=[1, 1, 1],
            relative_location=[20, 0, 0],
            relative_rotation=[0, 0, 0],
            move_action=[
                [ 1,  0],
                [ -0.3,  0],
                [ 0.5,  1],
                [ 0.5, -1],
                [ 0,  0]
            ],
            move_action_continuous={
                  "high": [ 1,  1],
                 "low":  [0, -1]
            }
        )
    @staticmethod
    def generate_motorbike_config(name, class_name, cam_id) -> dict:
        return dict(
            name=[name],
            cam_id=[cam_id],
            class_name=[class_name],
            internal_nav=True,
            scale=[1, 1, 1],
            relative_location=[20, 0, 0],
            relative_rotation=[0, 0, 0],
            move_action=[
                [ 1,  0],
                [ -0.3,  0],
                [ 0.5,  1],
                [ 0.5, -1],
                [ 0,  0]
            ],
            move_action_continuous={
                  "high": [ 1,  1],
                 "low":  [0, -1]
            }
        )
    @staticmethod
    def add_agent_type(agent_type):
        """Add a new agent type to the environment."""
        agent_config = {}
        if agent_type == 'player':
            agent_config[agent_type] = ConfigGenerator.create_player_config('bp_character_C_1', 'bp_character_C', 0)
            return agent_config
        elif agent_type == 'drone':
            agent_config[agent_type] = ConfigGenerator.create_drone_config('BP_drone01_C_1', 'BP_drone01_C', 0)
            return agent_config
        elif agent_type == 'animal':
            agent_config[agent_type] = ConfigGenerator.create_animal_config('BP_animal_C_1', 'BP_animal_C', 0)
            return agent_config
        elif agent_type == 'car':
            agent_config[agent_type] = ConfigGenerator.generate_car_config('BP_Hatchback_child_base_C_1', 'BP_Hatchback_child_base_C', 0)
            return agent_config
        elif agent_type == 'motorbike':
            agent_config[agent_type] = ConfigGenerator.generate_motorbike_config('BP_BaseBike_C_1', 'BP_BaseBike_C', 0)
            return agent_config
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

class RandomPopulationWrapper(Wrapper):
    def __init__(self, env, num_min=2, num_max=10, height_bias=0,agent_category=None, random_target=False, random_tracker=False):
        super().__init__(env)
        self.min_num = num_min
        self.max_num = num_max
        self.random_target_id = random_target
        self.random_tracker_id = random_tracker
        self.agent_category = agent_category
        self.height_bias = height_bias

    def step(self, action):
        obs, reward, termination,truncation, info = self.env.step(action)
        return obs, reward, termination,truncation, info

    def reset(self, **kwargs):
        env = self.env.unwrapped
        # env.target_agents = misc.convert_dict(env.target_configs)
        if not env.launched:  # we need to launch the environment
            env.launched = env.launch_ue_env(env.ue_log_path)
            env.init_agents()
            env.init_objects()
        else:
            env.init_agents()  # re-init agents to apply new configs
            env.init_objects()

        if self.min_num == self.max_num:
            env.num_agents = self.min_num
        else:
            # Randomize the number of agents
            env.num_agents = np.random.randint(self.min_num, self.max_num)
                
        if self.random_tracker_id:
            env.tracker_id = env.sample_tracker()
        if self.random_target_id:
            new_target = env.sample_target()
            if new_target != env.tracker_id:  # set target object mask to white
                env.unrealcv.build_color_dict(env.player_list)
                env.unrealcv.set_obj_color(env.player_list[env.target_id], env.unrealcv.color_dict[env.player_list[new_target]])
                env.unrealcv.set_obj_color(env.player_list[new_target], [255, 255, 255])
                env.target_id = new_target
        
        # check agent_type
        expect_agent_types = set(self.agent_category) if self.agent_category else set(env.agent_configs.keys())
        now_agent_types = set(env.agent_configs.keys())
        differ_agent_types = expect_agent_types - now_agent_types
        if differ_agent_types:  # if there are missing agent types
            # If there are missing agent types, we need to add them
            for agent_type in differ_agent_types:
                agent_config2add = ConfigGenerator.add_agent_type(agent_type)
                env.agent_configs = env.agent_configs | agent_config2add
                env.refer_agents = env.refer_agents | misc.convert_dict(agent_config2add)
        
        # add height bias and init target agents
        if hasattr(env, 'target_configs') and env.target_configs:
            self.add_height_bias(env, self.height_bias)
            env.target_agents = misc.convert_dict(env.target_configs)
            env.set_population(env.num_agents)

        states, info = self.env.reset(**kwargs)
        return states, info
    
    def add_height_bias(self, env, height_bias):
        for _, agent_config in env.target_configs.items():
            for i in range(len(agent_config["start_pos"])):
                agent_config["start_pos"][i][2] += height_bias
        return
