import numpy as np
from .agent_sample_agent_advanced import AgentBasedSamplerboost
import cv2
import os
import svgwrite
import json
from .placing_prompt import *
import re
class EventBasedAgentSampler(AgentBasedSamplerboost):
    def __init__(self, graph_pickle_file, model, config_path,bodyshot_path=None, name_dict_path=None):
        super().__init__(graph_pickle_file, model, config_path)
        self.cam_relative_height = 1200
        self.bodyshot_path = bodyshot_path
        if name_dict_path:
            with open(name_dict_path, 'r') as f:
                self.name_dict = json.load(f)
        self.available_actions = ["stand", "crouch","lie_down","enter_vehicle", "exit_vehicle", "pick_up","walk_to_point",]

    
    def event_planner(self,obs_rgb, shot_dict, available_actions):
        max_attempts = 5
        sys_prompt_event, usr_prompt_event = load_prompt("event_plot_prompt")
        name_list, body_shots = list(shot_dict.keys()), list(shot_dict.values())
        encoded_body_shots = [self.encode_image_array(shot) for shot in body_shots]
        encoded_obs_rgb = [self.encode_image_array(obs_rgb)]
        available_actions = ', '.join(available_actions)
        for attempt in range(max_attempts):
            try:
                response = self.vlm_plan(usr_prompt_event, sys_prompt_event, encoded_obs_rgb, encoded_body_shots, name_list, available_actions)
                event_description_match = re.search(r'<a>(.*?)</a>', response, re.DOTALL)
                action_dict_match = re.search(r'<b>(.*?)</b>', response, re.DOTALL)
                print("========event_description_match========\n", event_description_match)
                print("========action_dict_match========\n", action_dict_match)
            except Exception as e:
                print(f"尝试 {attempt+1}/{max_attempts}：发生错误 {e}，重新尝试...")
        return None, None

    def vlm_plan(self, user_prompt, sys_prompt, env_context, body_shots, names,available_actions):
        user_content_list = []
        user_content_list.append({
            "type": "text",
            "text": user_prompt,
        })
        
        for i, (name, base64_image) in enumerate(zip(names, body_shots)):
            prompt = f"{i+1} playable entity: {name}, full body shot:"
            user_content_list.append({
                "type": "text",
                "text": prompt,
            })
            user_content_list.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                }
            })
        
        user_content_list.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{env_context[0]}",
            }
        })
        user_content_list.append({
            "type": "text",
            "text": f"available actions for human figures: {available_actions}\n",
        })

        messages = [
            {"role": "system", "content": sys_prompt},
            
            {"role": "user", "content": user_content_list} 
        ]
        response = self.make_api_call(messages)
        return response

    def transition_camera_to_horizontal(self, env, cam_id, start_pos, end_height, steps=30, duration=2.0):
        import time
        
        start_z = start_pos[2]
        start_pitch = -90.0 
        
        
        target_z = self.ground_z + end_height 
        target_pitch = 0.0  # 水平向前
        
        # 3. 计算每一步的增量
        z_step = (target_z - start_z) / steps
        pitch_step = (target_pitch - start_pitch) / steps
        sleep_time = duration / steps
        
        print(f"开始相机运镜: 高度 {start_z:.1f}->{target_z:.1f}, 角度 {start_pitch}->{target_pitch}")
        
        current_z = start_z
        current_pitch = start_pitch
        
        # 4. 执行循环
        for i in range(steps):
            current_z += z_step
            current_pitch += pitch_step
            
            # 更新位置 (X, Y 保持不变)
            new_loc = [start_pos[0], start_pos[1], current_z]
            env.unrealcv.set_cam_location(cam_id, new_loc)
            
            # 更新旋转 (Yaw, Roll 保持不变，假设初始 Yaw=0)
            # 如果需要保持当前的 Yaw，可以先 get_cam_rotation 获取
            env.unrealcv.set_cam_rotation(cam_id, [current_pitch, 0, 0])
            
            time.sleep(sleep_time)
        
        env.unrealcv.set_cam_location(cam_id, [start_pos[0], start_pos[1], target_z])
        env.unrealcv.set_cam_rotation(cam_id, [target_pitch, 0, 0])
        print("相机运镜完成")

    def load_fullbody_shot(self, agent_configs):
        shot_dict = {}
        for type, config in agent_configs.items():
            for name, id in zip(config["name"],config["app_id"]):
                if type=='player' and id > 20:
                    type = 'robot_dog'
                reder_path = os.path.join(self.bodyshot_path, type, str(id), "0.png")
                img_array = cv2.imread(reder_path)
                true_name = self.name_dict[type][str(id)]
                shot_dict[true_name] = img_array
        return shot_dict


    def sample_agent_positions(self, env, agent_configs, cam_id=0, 
                               cam_count=3, vehicle_zones=None, height=800, **kwargs):
        # initial sampling
        session_state, object_list = self.prepare_sampling_session(
            env, agent_configs, vehicle_zones, cam_id, height, **kwargs
        )

        obs_rgb_small = cv2.resize(session_state['obs_rgb'], (session_state['obs_rgb'].shape[1] // 4, session_state['obs_rgb'].shape[0] // 4), interpolation=cv2.INTER_AREA)
        shot_dict = self.load_fullbody_shot(agent_configs)
        # all agent types
        agent_types = list(agent_configs.keys())
        available_actions = []
        if "car" not in agent_types and "motorbike" not in agent_types:
            available_actions = [act for act in self.available_actions if act not in ["enter_vehicle", "exit_vehicle"]]
            
        event_description, action_plan = self.event_planner(
            obs_rgb_small, shot_dict, available_actions
        )
        return

