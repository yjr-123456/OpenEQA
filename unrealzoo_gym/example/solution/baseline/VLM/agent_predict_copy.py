from unrealzoo_gym.example.solution.baseline.VLM.vlm_prompt import *
# from api import *
import os
import re
import argparse
import gym
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
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

client = OpenAI(api_key=api_key)

def call_api_vlm(sys_prompt, usr_prompt,base64_image):
    """
    Call the VLM API with the given prompt and image.
    """
    # Assuming OpenAI API is set up correctly
    response = client.chat.completions.create(
        model='gpt-4o-2024-08-06',
        max_tokens=10000,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {
                    "type": "text",
                    "text": usr_prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                }
                ]
            },
        ],

    )
    respon=  response.choices[0].message.content.strip()
    # print(f"[VLM RESPONSE] {respon}")
    return respon

def answer_question(sys_prompt, usr_prompt, key_frame, current_obs):
    """
    Call the VLM API with the given prompt and image.
    """
    response = client.chat.completions.create(
        model='gpt-4o-2024-08-06',
        max_tokens=10000,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {
                    "type": "text",
                    "text": usr_prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{current_obs}"}
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{key_frame}"}
                }
                ]
            },
        ],

    )
    return response.choices[0].message.content.strip()


def call_api_llm(sys_prompt,usr_prompt=None):
    """
    Call the LLM API with the given system prompt.
    """
    # Assuming OpenAI API is set up correctly
    response = client.chat.completions.create(
        model='gpt-3.5-turbo-0125',
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
    def __init__(self, k_frames = 3, memory_size= 5,con_th = 0.8,max_step = 50):
        self.target_list = []
        self.action = ([0,0], 0, 0)  # Default action: no movement
        self.phase = 0
        self.initialized = False
        self.final_answer = None  # To store the final answer after processing key frames
        # Initialize obs and info
        self.obs = None
        self.info = {}
        self.phase = 0
        self.con_th = con_th
        self.k_frames = k_frames  # Number of key frames to store
        self.max_step = max_step  # Maximum steps to take in the environment

        # Create buffers for actions and observations
        self.action_buffer = []
        self.obs_buffer = []
        self.confidence_buffer = []
        self.relevance_buffer = []
        self.memory_size = memory_size
        self.exploration_memory = deque(maxlen=memory_size)  # Memory for exploration phase
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
        self.answer = None

    def predict(self, obs, info):
        # Add a 1-second delay at the beginning of predict
        time.sleep(1)

        # Store the current observation and info
        self.obs = obs
        self.info = info
        self.obs_buffer.append(obs.copy())
        self.current_step += 1
        # Start the main logic chain if no pending actions
        if self.phase == 0:
            # Initialize by analyzing the question
            return self._handle_initial_phase()

        elif self.phase == 1:
            # Expore the environment to find clues for the question
            return self._handle_search_phase()

        elif self.phase ==2:
            return self._handle_answer_phase()

        # Default action if nothing else to do
        return ([0, 0], 0, 0)

    def _handle_initial_phase(self):
        sys_prompt = search_prompt_begin(self.question)
        usr_prompt = "Please analyze this question and find the target objects to search for in the environment."
        max_retries = 3
        for attempt in range(max_retries):
            try:
                res = call_api_llm(sys_prompt=sys_prompt,usr_prompt=usr_prompt)
                print(f"[Target Object] \n {res} \n\n\n")
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
        # current obs - question relavance
        relevance = self._quetion_image_relevance()
        # vlm give action
        sys_prompt = search_prompt(self.question, self._get_memory_context())
        usr_prompt = "Based on the question and the current observation, please give your action to take, and confidence to answer the question. The image is provided in base64 format."
        max_retries = 3
        for attempt in range(max_retries):
            try:
                res = call_api_vlm(sys_prompt=sys_prompt, usr_prompt=usr_prompt,base64_image=self.encode_image_array(self.obs))
                pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>\s*<c>(.*?)</c>\s*<d>(.*?)</d>', re.DOTALL | re.IGNORECASE)
                match = pattern.search(res)
                if match:
                    description = match.group(1).strip()
                    action_reasoning = match.group(2).strip()
                    action = match.group(3).strip()
                    confidence = float(match.group(4).strip())
                    self.confidence_buffer.append(confidence)
                    self.action_buffer.append(action)
                    self.exploration_memory.append({
                        'step': self.current_step,
                        'description': description,
                        'action_reasoning': action_reasoning,
                        'action': action,
                    })
                    print(f"[SEARCH PHASE] Step {self.current_step}\n Current Obs: {description}\n Action Reasoning: {action_reasoning}\n Action: {action}\n Confidence: {confidence}")
                    if confidence >= self.con_th or self.current_step >= self.max_step:
                        self.phase = 2  # Move to answer phase if confidence is high enough or reaches max steps
                    return self.action2action(action)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"[SEARCH] Failed after {max_retries} attempts: {str(e)}")
        return [-1]
    
    def _handle_answer_phase(self):
        relevance = self._quetion_image_relevance()
        if self.final_answer is not None:
            self.info['final_answer'] = self.final_answer
            print(f"[FINAL ANSWER] {self.final_answer}")
            return [-1]
        print("[ANSWER PHASE] Retrieving key frames and generating answer...")
        k = min(self.k_frames, len(self.obs_buffer))  # 最多检索5个关键帧
        if k == 0:
            self.final_answer = "I couldn't gather enough visual information to answer the question."
            self.info['final_answer'] = self.final_answer
            return [-1]
        
        key_frames = self._retrieve_top_k_frames(k)
        
        self.final_answer = self._generate_answer_from_key_frames(key_frames)
        print(f"[FINAL ANSWER] {self.final_answer}")
        self.info['final_answer'] = self.final_answer
        return [-1]
    
    def _retrieve_top_k_frames(self, k):        
        print(f"[RETRIEVAL] Selecting top {k} frames from {len(self.obs_buffer)} observations")
        print(len(self.obs_buffer), len(self.relevance_buffer))
        assert len(self.obs_buffer) == len(self.relevance_buffer)
        paired_data = list(zip(self.obs_buffer, self.relevance_buffer))
        sorted_pairs = sorted(
            paired_data, 
            key=lambda x: float(x[1]) if x[1] is not None else 0.0, 
            reverse=True
        )
        top_k_frames = []
        for i in range(min(k, len(sorted_pairs))):
            obs, relevance = sorted_pairs[i]
            top_k_frames.append(obs.copy())
            print(f"top {i+1} frame relevance: {relevance}")
        return top_k_frames

    def _quetion_image_relevance(self):
        sys_prompt = relavance_prompt(self.question)
        usr_prompt = "Based on the question and the current observation, please give the relevance of them ranging from 0 to 1.The image is provided in base64 format."
        max_retries = 5
        
        print(f"[RELEVANCE] Starting calculation (buffers: obs={len(self.obs_buffer)}, rel={len(self.relevance_buffer)})")
        
        for attempt in range(max_retries):
            try:
                res = call_api_vlm(sys_prompt=sys_prompt, usr_prompt=usr_prompt, base64_image=self.encode_image_array(self.obs))
                pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>', re.DOTALL | re.IGNORECASE)
                match = pattern.search(res)
                if match:
                    analyse = match.group(1).strip()
                    try:
                        relevance = float(match.group(2).strip())
                        relevance = max(0.0, min(1.0, relevance))
                        print(f"[RELEVANCE]:\n {analyse} - Relevance:\n {relevance}")
                        self.relevance_buffer.append(relevance) 
                        return relevance
                    except (ValueError, TypeError) as e:
                        print(f"[RELEVANCE] Float conversion error: {e}")
                        continue
                else:
                    print(f"[RELEVANCE] No XML match found in response: {res[:200]}...")
                    continue
                    
            except Exception as e:
                print(f"[RELEVANCE] API call failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print("[RELEVANCE] All attempts failed, using default relevance 0.2")
                    self.relevance_buffer.append(0.2)
                    return 0.2
                continue
        
        # ✅ 确保总是有返回值和buffer添加
        print("[RELEVANCE] Unexpected path, using default relevance 0.2")
        self.relevance_buffer.append(0.2)
        return 0.2
    def _generate_answer_from_key_frames(self, key_frames):    
        if not key_frames:
            return "No relevant information found."
        images = [frame for frame in key_frames]
        if len(images) == 1:
            combined_image = images[0]
        else:
            combined_image = self.concatenate_images(images)
    
        answer_prompt = question_answer_prompt(self.question, self._get_memory_context())
        usr_prompt = "Based on the question and the key frames, please provide a concise answer. The key frames are concated in one img which then transformed into base64 format."
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = answer_question(sys_prompt=answer_prompt, usr_prompt=usr_prompt,key_frame=self.encode_image_array(combined_image), current_obs=self.encode_image_array(self.obs))
                pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>', re.DOTALL | re.IGNORECASE)
                match = pattern.search(response)
                if match:
                    analyse = match.group(1).strip()
                    response = match.group(2).strip()
                else:
                    response = response.strip()
                answer = response
                self.answer = answer
                print(f"[ANSWER] Generated answer: {self.answer}")
                return answer
            except Exception as e:
                print(f"[ANSWER] Error generating answer (attempt {attempt+1}): {e}")
                if attempt == max_retries - 1:
                    return "I encountered an error while generating the answer."
        return "Unable to generate answer."

    def action2action(self, action):
        print(f"[ACTION] Received action: {action}")
        if action == "move_forward":
            return [0]
        elif action == "move_backward":
            return [1]
        elif action == "turn_left":
            return [2]
        elif action == "turn_right":
            return [3]
        else:
            print(f"[ACTION] Unrecognized action: {action}")
            return [-1]
    # def analyse_initial_image(self):
    #     prompt = initial_image_prompt()
    #     max_retries = 3
    #     for attempt in range(max_retries):
    #         try:
    #             res = call_api_vlm(sys_prompt=prompt, base64_image=self.encode_image_array(self.image_clue))
    #             self.person_text = res
    #             print(res)
    #             return True

    #         except Exception as e:
    #             if attempt == max_retries - 1:
    #                 raise ValueError(f"[IMAGE ANALYSE] Failed after {max_retries} attempts: {str(e)}")

    #     return False

    def _get_memory_context(self):
        if not self.exploration_memory:
            return "Start exploration phase, no history available."
        
        context_parts = []
        context_parts.append(f"Recent exploration history (last {len(self.exploration_memory)} steps):")
        
        for i, entry in enumerate(list(self.exploration_memory)[-3:], 1):  # 最近3步
            step_info = (
                f"Step {entry['step']}: \n"
                f"Observation Description: '{entry['description'][:100]}...' | "
                f"Reasoning: '{entry['action_reasoning'][:100]}...' | "
                f"Action: {entry['action']} | "
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

    def reset(self, question,obs, question_type='general',answer_list=None):
        print(f"[RESET] Resetting agent for new question: {question}")
        # reset question
        self.question_stem = question
        self.answer_list = answer_list if answer_list is not None else []
        self.question = question       # Store the question for processing
        if answer_list is not None:
            for ans in answer_list:
                self.question += f"\n{ans}"
        self.question += "\n"

        self.obs = obs
        self.info = {}
        
        
        self.phase = 0
        self.initialized = False
        self.current_step = 0
        
    
        self.target_list = []
        self.final_answer = None
        self.answer = None
        

        self.action_buffer = []
        self.obs_buffer = []
        self.confidence_buffer = []
        self.relevance_buffer = []
        
        self.action = [-1]

        print("[RESET] Agent reset complete. Ready for new question.")
        


