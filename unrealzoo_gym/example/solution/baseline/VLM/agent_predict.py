# from unrealzoo_gym.example.solution.baseline.VLM.vlm_prompt import *
from .vlm_prompt import *
import os
import re
import argparse
import cv2
import time
import numpy as np
import base64
from PIL import Image
import io
from openai import OpenAI
from datetime import datetime
from collections import deque
import random
from ultralytics import YOLO
import ast
from dotenv import load_dotenv
import sys

import logging
import os
from datetime import datetime
import sys
import json

def sanitize_filename(filename):
    """清理文件名中的非法字符"""
    # Windows禁用字符: < > : " | ? * \ /
    # 替换为下划线
    sanitized = re.sub(r'[<>:"|?*\\/]', '_', filename)
    # 移除首尾空格和点号
    sanitized = sanitized.strip('. ')
    # 限制长度
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    return sanitized


def setup_logging(logger_base_dir="logs", env_name=None, question_type=None, batch_id=None, model="default_model"):
    """
    按环境名和问题类型创建日志文件
    """
    # 创建日志目录（支持多级目录）
            
    # if logger_base_dir:
    #     log_base_dir = f"{logger_base_dir}/experiment_results/logs/{model}"
    # else:
    #     base_dir = os.path.dirname(os.path.abspath(__file__))
    #     log_base_dir = os.path.join(base_dir, "experiment_results","logs", model)
    # os.makedirs(log_base_dir, exist_ok=True)        
    # if env_name and question_type and batch_id:
    # # 按环境名和问题类型创建子目录
    #     log_dir = os.path.join(log_base_dir, env_name, question_type, batch_id)
    # elif env_name and question_type:
    #     log_dir = os.path.join(log_base_dir, env_name, question_type)
    # elif env_name:
    #     log_dir = os.path.join(log_base_dir, env_name)
    # else:
    #     log_dir = log_base_dir
    log_dir = logger_base_dir
    os.makedirs(log_dir, exist_ok=True)
    safe_env_name = sanitize_filename(env_name) if env_name else "unknown"
    safe_question_type = sanitize_filename(question_type) if question_type else "general"
    safe_batch_id = sanitize_filename(batch_id) if batch_id else None
    
    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # if env_name and question_type and batch_id:
    #     log_filename = os.path.join(log_dir, f"agent_{safe_env_name}_{safe_question_type}_{safe_batch_id}_{timestamp}.log")
    #     logger_name = f"agent_{safe_env_name}_{safe_question_type}_{safe_batch_id}"
    # elif env_name and question_type:
    #     log_filename = os.path.join(log_dir, f"agent_{safe_env_name}_{safe_question_type}_{timestamp}.log")
    #     logger_name = f"agent_{safe_env_name}_{safe_question_type}"
    # elif env_name:
    #     log_filename = os.path.join(log_dir, f"agent_{safe_env_name}_{timestamp}.log")
    #     logger_name = f"agent_{safe_env_name}"
    # else:
    log_filename = os.path.join(log_dir, f"agent_{model}_{timestamp}.log")
    logger_name = f"agent_{model}"
    
    # 创建独立的logger实例（避免不同环境间冲突）
    logger = logging.getLogger(logger_name)
    
    # 清除现有处理器
    if logger.handlers:
        logger.handlers.clear()
    
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 防止向上传播到根logger（避免重复输出）
    logger.propagate = False
    
    logger.info(f"日志系统已初始化: {env_name or 'Unknown'}/{question_type or 'General'}")
    logger.info(f"日志文件: {log_filename}")
    
    return logger
logger = None
load_dotenv(override=True) 
import queue

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
    
    # 从环境变量获取API密钥
    api_key = os.environ.get(model_config["api_key_env"])
    if not api_key:
        raise ValueError(f"环境变量 {model_config['api_key_env']} 未设置")
    
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

def call_api_vlm(sys_prompt, usr_prompt,base64_image_list):
    """
    Call the VLM API with the given prompt and image.
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
    # print(f"[VLM RESPONSE] {respon}")
    return respon

def action_planner(sys_prompt, usr_prompt, last_action, mem_info ,base64_image_list):
    """
    Call the VLM API with the given prompt and image.
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
    if mem_info:
        messages.append({"role": "user", "content": [
            {
                "type": "text",
                "text": f"exploration memory: {mem_info}",
            }
        ]})
    if len(last_action) != 0:
        assert len(last_action) == len(base64_image_list), \
            f"Length mismatch: last_action={len(last_action)}, base64_image_list={len(base64_image_list)}"
        for i, (base64_image, action) in enumerate(zip(base64_image_list, last_action)):
            prompt = f"last {len(last_action) - i} action: {action}, observation:"
            messages.append({
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        }
                    }
                ]
            })
    else:
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
        model= current_model_name,
        max_tokens=10000,
        messages=messages
    )
    respon=  response.choices[0].message.content.strip()
    # print(f"[VLM RESPONSE] {respon}")
    return respon

def answer_question(sys_prompt, usr_prompt, clue_list, rgb_base64_image_list, depth_base64_image_list):
    """
    Call the VLM API with the given prompt and image.
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
    for i, (rgb_base64_image, depth_base64_image, clue_info) in enumerate(zip(rgb_base64_image_list, depth_base64_image_list, clue_list)):
        prompt = f"Key Frame {i+1}(rgb observation and depth observation) - Relevance: {clue_info['relevance']:.3f}"
        messages.append({
            "role": "user", 
            "content": [
                {   
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{rgb_base64_image}",
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{depth_base64_image}",
                    }
                },
                {
                    "type": "text",
                    "text": prompt,
                }
            ]
        })
    
    response = client.chat.completions.create(
        model=current_model_name,
        max_tokens=10000,
        messages=messages
    )
    
    return response.choices[0].message.content.strip()

def call_api_llm(sys_prompt,usr_prompt=None):
    """
    Call the LLM API with the given system prompt.
    """
    # Assuming OpenAI API is set up correctly
    response = client.chat.completions.create(
        model= current_model_name,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {
                    "type": "text",
                    "text": usr_prompt,
                }
                ]
            },
        ],

    )
    return response.choices[0].message.content.strip()

class agent:
    def __init__(self, k_frames = 3, max_action_length=3, memory_size= 5,con_th = 0.8,max_step = 50, model= "doubao",config_path="model_config.json"):

        self.model = model
        initialize_model(model, f"{config_path}/model_config.json")

        self.target_list = []
        self.action = queue.deque(maxlen=max_action_length)  # Default action: no movement
        self.phase = 0
        self.initialized = False
        self.final_answer = None  # To store the final answer after processing key frames
        # Initialize obs and info
        self.info = {}
        self.phase = 0
        self.con_th = con_th
        self.k_frames = k_frames  # Number of key frames to store
        self.max_step = max_step  # Maximum steps to take in the environment
        self.max_action_length = max_action_length  # Maximum length of action sequences


        # Create buffers for actions and observations
        self.action_buffer = []
        self.obs_buffer = []
        self.depth_buffer = []
        self.confidence_buffer = []
        self.relevance_buffer = []
        self.clue_buffer = []
        self.memory_size = memory_size
        self.exploration_memory = []  # Memory for exploration phase
        self.memory_stats = {
            "current_analyse:": "",
            "action": "",
            "confidence": 0.0
        }
        # step counter
        self.current_step = 0
        # Initialize question and answer
        self.question = None
        self.question_stem = None


    def predict(self, obs_rgb, obs_depth, info):
        # Add a 1-second delay at the beginning of predict
        time.sleep(1)

        # Store the current observation and info
        self.obs_rgb = obs_rgb
        self.obs_depth, _ = self.convert_depth_to_8bit(obs_depth, method='inverse')
        self.info = info
        self.current_step += 1
        # Start the main logic chain if no pending actions
        if hasattr(self, 'termination') and self.termination:
            logger.info("[PREDICT] Episode terminated due to high confidence")
            return [-1]
    
        if hasattr(self, 'truncation') and self.truncation:
            logger.info("[PREDICT] Episode truncated due to max steps")
            return [-1]


        if self.phase == 0:
            # recognize main obj and other objects in the question
            return self._handle_initial_phase()

        elif self.phase == 1:
            # search for target objects
            return self._handle_search_phase()
        
        elif self.phase ==2:
            # navigate to target objects
            return [-1]
        
        # Default action if nothing else to do
        return [-1]

    def _handle_initial_phase(self):
        sys_prompt = search_prompt_begin(self.question)
        usr_prompt = "Please analyze this question and find the target objects to search for in the environment."
        max_retries = 3
        self.curr_actions = ["stay_still"]
        for attempt in range(max_retries):
            try:
                res = call_api_llm(sys_prompt=sys_prompt,usr_prompt=usr_prompt)
                logger.info(f"[Target Object] \n {res} \n\n\n")
                pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>', re.DOTALL | re.IGNORECASE)
                match = pattern.search(res)
                if match:
                    analyse = match.group(1).strip()
                    target_obj = match.group(2).strip()
                    if target_obj != []:
                        self.target_list = ast.literal_eval(target_obj)
                        self.phase = 1
                        return self._handle_search_phase()
                    else:
                        self.phase = 1
                        return self._handle_search_phase()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"[SEARCH] Failed after {max_retries} attempts: {str(e)}")
        # Return default action if already initialized
        return [-1]

    def _handle_search_phase(self):
        self.obs_buffer.append(self.obs_rgb.copy())
        self.depth_buffer.append(self.obs_depth.copy())
        # relevance
        relevance = self._question_image_relevance()
        
        # collect clues
        # depth_flag = self.ask_need_for_depth()
        clue_flag, clue = self._clue_collection(True)
        self.clue_buffer.append({
            "relevance": relevance,
            "clue_flag": clue_flag,
            "clue": clue
        })
        
        # answer question
        if len(self.obs_buffer) >= 10 or len(self.curr_actions) == 0:
            self.answer_candidate.append((self._handle_answer_phase(), self.info['confidence']))
            if self.info["confidence"] >= self.con_th:
                self.termination = True
                self.final_answer = self.answer_candidate[-1][0]
                logger.info(f"[FINAL ANSWER] {self.final_answer} with confidence {self.info['confidence']}, True answer: {self.true_answer}")
                return [-1]
            if self.current_step >= self.max_step:
                self.truncation = True
                if self.answer_candidate:
                    sorted_candidates = sorted(self.answer_candidate, key=lambda x: float(x[1]), reverse=True)
                    self.final_answer = sorted_candidates[0][0]
                    logger.info(f"[Truncation][FINAL ANSWER] {self.final_answer} with confidence {self.info['confidence']}, True answer: {self.true_answer}")
                return [-1]
        
        # make next action
        if len(self.action) == 0:
            actions = self._action_planning()
            self.curr_actions = actions
            if len(actions) == 0:
                actions = ["stay_still"]
            # add to action buffer
            for action in actions:
                if len(self.action) <= self.max_action_length:
                    self.action.append(action)
        action = self.action.popleft()
        self.action_buffer.append(action)
        return self.action2action(action)
    
    def _handle_answer_phase(self):
        # if self.final_answer is not None:
        #     self.info['final_answer'] = self.final_answer
        #     print(f"[FINAL ANSWER] {self.final_answer}")
        #     return self.final_answer
        logger.info("[ANSWER PHASE] Retrieving key frames and generating answer...")
        k = min(self.k_frames, len(self.obs_buffer))  # 最多检索5个关键帧
        if k == 0:
            answer = "I couldn't gather enough visual information to answer the question."
            return answer
        
        key_frames = self._retrieve_top_k_frames(k)
        
        answer = self._generate_answer_from_key_frames(key_frames)
        # print(f"[CURRENR ANSWER] {answer}")
        # self.info['final_answer'] = self.final_answer
        return answer

    def ask_need_for_depth(self):
        sys_prompt = ask_for_depth(self.question, self.target_list)
        usr_prompt = "Based on the question, please answer yes or no."
        max_retries = 5
        logger.info(f"[DEPTH] Starting depth necessity analysis")
        for attempt in range(max_retries):
            try:
                res = call_api_vlm(sys_prompt=sys_prompt, usr_prompt=usr_prompt, base64_image_list=[self.encode_image_array(self.obs_rgb), self.encode_image_array(self.obs_depth)])
                pattern = re.compile(r'<a>(.*?)</a>', re.DOTALL | re.IGNORECASE)
                match = pattern.search(res)
                if match:
                    result = match.group(1).strip().lower()
                    if "yes" in result:
                        logger.info(f"[DEPTH] Depth needed based for current clues collection: {result}")
                        return True
                    else:
                        logger.info(f"[DEPTH] No depth needed for current clues collection: {result}")
                        return False
                else:
                    logger.info(f"[DEPTH] No XML match found in response: {res[:200]}...")
                    continue
                    
            except Exception as e:
                logger.info(f"[DEPTH] API call failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.info("[DEPTH] All attempts failed, using default clue 'none'")
                    return False
                continue
        logger.info("[DEPTH] Unexpected path, using default clue 'none'")
        return False

    def _retrieve_top_k_frames(self, k):        
        logger.info(f"[RETRIEVAL] Selecting top {k} frames from {len(self.obs_buffer)} observations")
        logger.info(f"Buffer lengths - obs: {len(self.obs_buffer)}, clue: {len(self.clue_buffer)}")
        
        assert len(self.obs_buffer) == len(self.clue_buffer), \
            f"Buffer length mismatch: obs={len(self.obs_buffer)}, clue={len(self.clue_buffer)}"
        
        paired_data = list(zip(self.obs_buffer, self.depth_buffer, self.clue_buffer, range(len(self.obs_buffer))))
        
        clue_frames = []
        no_clue_frames = []
        
        for rgb_obs, depth_obs, clue_data, idx in paired_data:
            frame_info = {
                'rgb_obs': rgb_obs,
                'depth_obs': depth_obs,
                'clue_data': clue_data,
                'index': idx,
                'relevance': clue_data.get('relevance', 0.0),
                'clue_flag': clue_data.get('clue_flag', 0),
                'clue': clue_data.get('clue', 'none')
            }
            
            if clue_data.get('clue_flag', 0) == 1:
                clue_frames.append(frame_info)
            else:
                no_clue_frames.append(frame_info)
        
        # 分别对两组按相关性排序
        clue_frames.sort(key=lambda x: float(x['relevance']) if x['relevance'] is not None else 0.0, reverse=True)
        no_clue_frames.sort(key=lambda x: float(x['relevance']) if x['relevance'] is not None else 0.0, reverse=True)
        
        logger.info(f"[RETRIEVAL] Available frames: {len(clue_frames)} with clues, {len(no_clue_frames)} without clues")
        
        # 优先选择有线索的帧，不足时补充高相关性的无线索帧
        selected_frames = []
        selected_count = 0
        
        # 首先选择所有有线索的帧（最多k个）
        for frame_info in clue_frames:
            if selected_count < k:
                selected_frames.append(frame_info)
                selected_count += 1
                logger.info(f"  Selected clue frame {selected_count}: Relevance {frame_info['relevance']:.3f}, Clue: {frame_info['clue']}")
        
        # 如果还需要更多帧，从无线索帧中选择高相关性的
        for frame_info in no_clue_frames:
            if selected_count < k:
                selected_frames.append(frame_info)
                selected_count += 1
                logger.info(f"  Selected no-clue frame {selected_count}: Relevance {frame_info['relevance']:.3f}")
        
        # 提取图像数据
        top_k_frames = [(frame_info['rgb_obs'].copy(),frame_info['depth_obs'].copy(), frame_info['index']) for frame_info in selected_frames]

        logger.info(f"[RETRIEVAL] Final selection: {len(top_k_frames)} frames")
        return top_k_frames

    def _question_image_relevance(self):
        sys_prompt = relavance_prompt(self.question)
        usr_prompt = "Based on the question and the current observation, please give the relevance of them ranging from 0 to 1.If there are no The image is provided in base64 format."
        max_retries = 5
        logger.info(f"[RELEVANCE] Starting calculation (buffers: obs={len(self.obs_buffer)}, rel={len(self.relevance_buffer)})")
        
        for attempt in range(max_retries):
            try:
                res = call_api_vlm(sys_prompt=sys_prompt, usr_prompt=usr_prompt, base64_image_list=[self.encode_image_array(self.obs_rgb)])
                pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>', re.DOTALL | re.IGNORECASE)
                match = pattern.search(res)
                if match:
                    analyse = match.group(1).strip()
                    try:
                        relevance = float(match.group(2).strip())
                        relevance = max(0.0, min(1.0, relevance))
                        logger.info(f"[RELEVANCE]:\n {analyse} - Relevance:\n {relevance}")
                        # self.relevance_buffer.append(relevance) 
                        return relevance
                    except (ValueError, TypeError) as e:
                        logger.info(f"[RELEVANCE] Float conversion error: {e}")
                        continue
                else:
                    logger.info(f"[RELEVANCE] No XML match found in response: {res[:200]}...")
                    continue
                    
            except Exception as e:
                logger.info(f"[RELEVANCE] API call failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.info("[RELEVANCE] All attempts failed, using default relevance 0.2")
                    
                    return 0.2
                continue
        logger.info("[RELEVANCE] Unexpected path, using default relevance 0.2")
        return 0.2

    def _clue_collection(self, depth_flag):
        sys_prompt = key_clue_collection(self.question, self.target_list)
        usr_prompt = "Based on the question and the current observation, please collect the clues for the question.If there are no The image is provided in base64 format."
        max_retries = 5
        
        logger.info(f"[CLUES] Starting collecting clues")
        
        for attempt in range(max_retries):
            try:
                base64_image_list=[self.encode_image_array(self.obs_rgb)]
                if depth_flag:
                    base64_image_list.append(self.encode_image_array(self.obs_depth))
                res = call_api_vlm(sys_prompt=sys_prompt, usr_prompt=usr_prompt, base64_image_list=base64_image_list)
                pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>', re.DOTALL | re.IGNORECASE)
                match = pattern.search(res)
                if match:
                    analyse = match.group(1).strip()
                    try:
                        clue = match.group(2).strip()
                        
                        logger.info(f"[CLUES]:\n {analyse} - CLUES: {clue}")
                        if clue.lower() == "none":
                            return 0, "none"
                        else:

                            return 1, clue
                    except (ValueError, TypeError) as e:
                        logger.info(f"[CLUES] Float conversion error: {e}")
                        continue
                else:
                    logger.info(f"[CLUES] No XML match found in response: {res[:200]}...")
                    continue
                    
            except Exception as e:
                logger.info(f"[CLUES] API call failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.info("[CLUES] All attempts failed, using default clue 'none'")
                    return 0, "none"
                continue
        logger.info("[CLUES] Unexpected path, using default clue 'none'")
        return 0, "none"
    
    def _action_planning(self):
        # vlm give action
        sys_prompt = direction_planner(self.question, self.target_list)
        usr_prompt = "Based on the information we provided you, please give your actions to take.The image is provided in base64 format."
        max_retries = 3
        mem_info = ""
        if self.exploration_memory:
            for i, entry in enumerate(list(self.exploration_memory)[-1:], 1):  # 最近1步
                mem_info = (
                    f"Area of interest analysis: '{entry['intersting_area_reasoning'][:]}...' | "
                    f"Action Reasoning: '{entry['action_reasoning'][:]}...' | "
                )
        base64_image_list =[]
        last_action = self.exploration_memory[-1]['action_list'] if self.exploration_memory else []
        action_length = len(last_action)
        if action_length != 0:
            for image in self.obs_buffer[-action_length:]:
                base64_image_list.append(self.encode_image_array(image))
        else:
            base64_image_list.append(self.encode_image_array(self.obs_rgb))
        for attempt in range(max_retries):
            try:
                res = action_planner(sys_prompt=sys_prompt, usr_prompt=usr_prompt, last_action=last_action, mem_info=mem_info, base64_image_list=base64_image_list)
                pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>\s*<c>(.*?)</c>', re.DOTALL | re.IGNORECASE)
                match = pattern.search(res)
                if match:
                    area_reasoning = match.group(1).strip()
                    action_reasoning = match.group(2).strip()
                    action = ast.literal_eval(match.group(3).strip())
                    self.exploration_memory.append({
                        "intersting_area_reasoning": area_reasoning,
                        "action_reasoning": action_reasoning,
                        "action_list": action,
                        "action_length": len(action),
                    })
                    logger.info(f"[Area Reasoning]: {area_reasoning} \n [Action Reasoning]: {action_reasoning}\n [ACTION PLANNING] Action List: {action} \n")
                    return action
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.info(f"[ACTION PLANNING] Exceed max action generation attempt (attempt {attempt+1}")
                    return ["stay_still"]  # Default action if all attempts fail
                
        return ["stay_still"]  # Default action if no valid action is found

    # def _reverse_juge(self):

    def _generate_answer_from_key_frames(self, key_frames):    
        if not key_frames:
            self.info['confidence'] = 0.0
            return "No relevant information found."
        rgb_images = [self.encode_image_array(frame[0]) for frame in key_frames]
        depth_images = [self.encode_image_array(frame[1]) for frame in key_frames]
        clue_list = [self.clue_buffer[frame[2]] for frame in key_frames]
        # if len(images) == 1:
        #     combined_image = images[0]
        # else:
        #     combined_image = self.concatenate_images(images)

        answer_prompt = question_answer_prompt(self.question, self.target_list)
        usr_prompt = "Based on the question and the key frames, please provide a concise answer. The key frames are concated in one img which then transformed into base64 format."
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = answer_question(sys_prompt=answer_prompt, usr_prompt=usr_prompt, clue_list=clue_list,rgb_base64_image_list=rgb_images, depth_base64_image_list=depth_images)
                pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>\s*<c>(.*?)</c>', re.DOTALL | re.IGNORECASE)
                match = pattern.search(response)
                if match:
                    analyse = match.group(1).strip()
                    answer = match.group(2).strip()
                    confidence = float(match.group(3).strip())
                    self.info['confidence'] = confidence
                    self.info['question_analyse'] = analyse
                    self.info['question_answer'] = answer
                    logger.info(f"[ANSWER] Current generated answer: {answer}")
                    logger.info(f"[ANSWER] Analysis: {analyse}")
                    return answer
                else:
                    answer = response.strip()
                    self.info['confidence'] = 0.3  # 低置信度
                    self.info['question_answer'] = answer
                    answer = response.strip()
                    logger.info(f"[ANSWER] Current generated answer (no pattern): {answer}")
                    return answer
            except Exception as e:
                logger.info(f"[ANSWER] Error generating answer (attempt {attempt+1}): {e}")
                if attempt == max_retries - 1:
                    self.info['confidence'] = 0.1
                    return "I encountered an error while generating the answer."
        self.info['confidence'] = 0.1
        return "Unable to generate answer."

    def action2action(self, action):
        logger.info(f"[ACTION] Received action: {action}")
        if action == "move_forward":
            return [0]
        elif action == "move_backward":
            return [1]
        elif action == "turn_left":
            return [2]
        elif action == "turn_right":
            return [3]
        elif action == "stay_still":
            return [-1]
        else:
            logger.info(f"[ACTION] Unrecognized action: {action}")
            return [-1]
    
    def _get_memory_context(self):
        if not self.exploration_memory:
            return "Start exploration phase, no history available."
        
        context_parts = []
        context_parts.append(f"Recent exploration history (last {len(self.exploration_memory)} steps):")

        for i, entry in enumerate(list(self.exploration_memory)[-1:], 1):  # 最近1步
            step_info = (
                f"Area of interest analysis: '{entry['intersting_area_reasoning'][:]}...' | "
                f"Action Reasoning: '{entry['action_reasoning'][:]}...' | "
                f"Action List: {entry['action_list']} | "
            )
            context_parts.append(step_info)
        
        return "\n".join(context_parts)

    def encode_image_array(self, image_array):
        # Convert the image array to a PIL Image object
        image = Image.fromarray(np.uint8(image_array))

        # Save the PIL Image object to a bytes buffer
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")

        # Encode the bytes buffer to Base64
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return img_str

    def convert_depth_to_8bit(
        self,
        depth_array, 
        method='linear',
        min_val=None,
        max_val=None,
        invert=False,
        gamma=1.0,
        log_scale=False,
        clip_percentile=None
    ):

        import numpy as np
        
        # 复制数组防止修改原数据
        depth = depth_array.copy()
        
        # 保存原始信息
        original_min = float(depth.min())
        original_max = float(depth.max())
        
        # 应用倒数处理
        if method == 'inverse':
            # 避免除零
            depth = 1.0 / (depth + 1e-8)
        
        # 过滤无效值
        valid_mask = ~np.isnan(depth) & ~np.isinf(depth) & (depth > 0)
        valid_min = depth[valid_mask].min() if valid_mask.any() else 0
        valid_max = depth[valid_mask].max() if valid_mask.any() else 1
        
        # 使用指定的范围或有效数据范围
        min_val = valid_min if min_val is None else min_val
        max_val = valid_max if max_val is None else max_val
        
        # 应用百分比截断
        if clip_percentile is not None:
            low, high = clip_percentile
            if valid_mask.any():
                p_low = np.percentile(depth[valid_mask], low)
                p_high = np.percentile(depth[valid_mask], high)
                depth = np.clip(depth, p_low, p_high)
                min_val = p_low
                max_val = p_high
        
        # 应用对数缩放
        if log_scale and valid_mask.any():
            depth[valid_mask] = np.log1p(depth[valid_mask] - min_val + 1e-8)
            max_log = np.log1p(max_val - min_val + 1e-8)
            depth = depth / max_log
        else:
            # 线性归一化
            depth_range = max_val - min_val
            if depth_range > 0:
                depth = (depth - min_val) / depth_range
            else:
                depth = np.zeros_like(depth)
        
        # 应用gamma校正
        if gamma != 1.0:
            depth = np.power(depth, gamma)
        
        # 反转（如果需要）
        if invert:
            depth = 1.0 - depth
        
        # 转换为8位
        depth_8bit = (depth * 255).astype(np.uint8)
        
        # 创建元数据
        metadata = {
            'original_min': original_min,
            'original_max': original_max,
            'process_min': float(min_val),
            'process_max': float(max_val),
            'method': method,
            'gamma': gamma,
            'log_scale': log_scale,
            'invert': invert,
            'clip_percentile': clip_percentile
        }
        
        return depth_8bit, metadata

    def concatenate_images(self, image_list):
        height, width, channels = image_list[0].shape

        total_width = width * len(image_list)
        concatenated_image = np.zeros((height, total_width, channels), dtype=np.uint8)

        for i, img in enumerate(image_list):
            concatenated_image[:, i * width:(i + 1) * width, :] = img

        return concatenated_image

    def add_vertical_lines(self, image_array):
        h, w, c = image_array.shape

        line1 = w // 3
        line2 = w * 2 // 3
        line_color = (0, 0, 255)
        line_thickness = 2

        # Create a copy to avoid modifying the original
        image_copy = image_array.copy()
        cv2.line(image_copy, (line1, 0), (line1, h), line_color, line_thickness)
        cv2.line(image_copy, (line2, 0), (line2, h), line_color, line_thickness)
        return image_copy

    def reset(self, question, obs_rgb, obs_depth, target_type, 
              question_type='general', answer_list=None, batch_id=None,
              question_answer=None, env_name=None, logger_base_dir=None):
        global logger
        
        # 为每个环境和问题类型组合创建独立的日志文件
        if logger is None or hasattr(self, '_current_env_type') and self._current_env_type != (env_name, question_type):
            # 保存当前环境和问题类型组合
            self._current_env_type = (env_name, question_type)
                        
            # 重新设置日志系统
            logger = setup_logging(
                logger_base_dir=logger_base_dir,
                env_name=env_name,
                question_type=question_type,
                batch_id=batch_id,
                model=self.model
            )
        
        logger.info(f"[RESET] Resetting agent for new question: {question}")
        logger.info(f"  Environment: {env_name}")
        logger.info(f"  Question Type: {question_type}")
        logger.info(f"  Batch ID: {batch_id}")
        # reset question
        self.question_stem = question
        self.answer_list = answer_list if answer_list is not None else []
        self.question = question       # Store the question for processing
        if answer_list is not None:
            for ans in answer_list:
                self.question += f"\n{ans}"
        self.question += "\n"
        self.true_answer = question_answer if question_answer is not None else None
        self.target_type = target_type
        self.target_type = target_type.copy() if isinstance(target_type, list) else target_type
        if isinstance(self.target_type, list) and "player" in self.target_type:
            index = self.target_type.index("player")
        
        self.target_type[index] = "person"
        self.obs_rgb = obs_rgb
        self.obs_depth, _ = self.convert_depth_to_8bit(obs_depth, method='inverse')
        self.info = {}
        
        
        self.phase = 0
        self.initialized = False
        self.current_step = 0
        self.action = queue.deque(maxlen=self.max_action_length)
    
        self.target_list = []
        self.final_answer = None
        self.exploration_memory = [] 

        self.action_buffer = []
        self.obs_buffer = []
        self.depth_buffer = []
        self.confidence_buffer = []
        self.relevance_buffer = []
        self.clue_buffer = []

        self.termination = False
        self.truncation = False
        self.answer_candidate = []
        


