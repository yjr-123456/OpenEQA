import argparse

from torch.utils.tensorboard.summary import draw_boxes

# import gym_rescue
import gym
from gym import wrappers
import cv2
import time
import numpy as np
# from gym_rescue.envs.wrappers import time_dilation, early_done, monitor, configUE, task_cue
import os
import json
from unrealzoo_gym.example.solution.baseline.VLM.agent_predict_copy import agent
# from VLM_Agent.Rough_agent import agent
from ultralytics import YOLO
import torch
import base64
# from trajectory_visualizer import TrajectoryVisualizer

class HybridAgent:
    def __init__(self,reference_text,reference_image):
        # Initialize VLM agent
        self.vlm_agent = agent(reference_text,reference_image)
        
        # Initialize detection model
        self.yolo_model = YOLO('checkpoints/yolo11x.pt')
        
        # State tracking
        self.person_detected = False
        self.stretcher_detected = False
        self.current_target = None  # 'person' or 'stretcher'
        
        self.foreward = ([0,50],0,0)

        self.backward = ([0,-50],0,0)
        self.turnleft = ([-20,0],0,0)
        self.turnright = ([20,0],0,0)
        self.carry = ([0,0],0,3)
        self.drop = ([0,0],0,4)
        self.noaction = ([0,0],0,0)
    def reset(self, text, image):
        self.vlm_agent.reset(text, image)
        self.person_detected = False
        self.stretcher_detected = False
        self.current_target = None


    def draw_bbox_on_obs(self,obs, boxes, labels, color=(0, 255, 0), thickness=2):
        """
        Draw bounding boxes on the observation image.

        Args:
            obs: The observation image (numpy array).
            boxes: List of bounding boxes, each in the format [x, y, w, h].
            labels: List of labels corresponding to the bounding boxes.
            color: Color of the bounding box (default: green).
            thickness: Thickness of the bounding box lines (default: 2).
        """
        for box, label in zip(boxes, labels):
            x, y, w, h = box
            top_left = (int(x - w / 2), int(y - h / 2))
            bottom_right = (int(x + w / 2), int(y + h / 2))
            cv2.rectangle(obs, top_left, bottom_right, color, thickness)
            cv2.putText(obs, label, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return obs

    def predict(self, obs, info):
        # First try to detect objects using YOLO
        if info['picked'] == 1:
            results = self.yolo_model(source=obs, imgsz=640, conf=0.1)
        else:
            results = self.yolo_model(source=obs, imgsz=640, conf=0.4)

        boxes = results[0].boxes  # get all detected bounding box

        boxes_tmp = [box.xywh[0].tolist() for box in boxes]
        labels_tmp = [self.yolo_model.names[int(box.cls.item())] for box in results[0].boxes]

        # Draw bounding boxes on the observation image
        obs_with_bbox = self.draw_bbox_on_obs(obs, boxes_tmp, labels_tmp)
        cv2.imshow('Observation with BBox', obs_with_bbox)
        cv2.waitKey(1)
        # Check for person and stretcher in detections
        person_box = None
        stretcher_box = None
        truck_box=None
        if info['picked']==1:
            self.person_detected=True
        for box in boxes:
            cls = int(box.cls.item())
            if self.yolo_model.names[cls] == 'person' and not self.person_detected:
                person_box = box.xywh[0].tolist()
            elif self.yolo_model.names[cls] == 'suitcase' and self.person_detected:
                stretcher_box = box.xywh[0].tolist()
            # elif self.yolo_model.names[cls] == 'bench'  and self.person_detected:
            #     stretcher_box = box.xywh[0].tolist()
            elif self.yolo_model.names[cls] =='truck'and self.person_detected:
                truck_box = box.xywh[0].tolist()
            elif self.yolo_model.names[cls] =='bus'and self.person_detected:
                truck_box = box.xywh[0].tolist()

        # If we have a detection, use detection-based movement
        if person_box and not self.person_detected:
            return self._move_based_on_detection(person_box, 'person')
        elif stretcher_box and self.person_detected:
            return self._move_based_on_detection(stretcher_box, 'stretcher')
        elif truck_box and self.person_detected:
            return self._move_based_on_detection(truck_box, 'truck')


        # If no detection, use VLM agent for exploration
        return self.vlm_agent.predict(obs, info)

    def _move_based_on_detection(self, box, target_type):
        x0, y0, w_, h_ = box
        
        if target_type == 'person':
            # if w_ > h_:
            if y0 - 0.5*h_ > 350 and x0>220 and x0<420:
                # self.person_detected = True
                return self.carry
            else:
                if x0 < 220:
                    return self.turnleft
                elif x0 > 420:
                    return self.turnright
                else:
                    return self.foreward
        elif target_type =='stretcher':
            if y0 - 0.5*h_ > 350  and x0>220 and x0<420 :
                return self.drop
            else:
                if x0 < 220:
                    return self.turnleft
                elif x0 > 420:
                    return self.turnright
                else:
                    return self.foreward
        elif target_type == 'truck':  # stretcher
            if w_> 220 and h_>220  and x0>220 and x0<420 :
                return self.drop
            else:
                if x0 < 220:
                    return self.turnleft
                elif x0 > 420:
                    return self.turnright
                else:
                    return self.foreward

