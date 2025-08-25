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
import os
# import torch
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, agents, augmentation, configUE
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import tqdm
# class RandomAgent(object):
#     """The world's simplest agent!"""
#     def __init__(self, action_space):
#         self.action_space = action_space
#         self.count_steps = 0
#         self.action = self.action_space.sample()

#     def act(self, observation, keep_steps=10):
#         self.count_steps += 1
#         if self.count_steps > keep_steps:
#             self.action = self.action_space.sample()
#             self.count_steps = 0
#         else:
#             return self.action
#         return self.action

#     def reset(self):
#         self.action = self.action_space.sample()
#         self.count_steps = 0


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space
        self.count_steps = 0
        
        # list type action_space
        if isinstance(self.action_space, list):
            # if list, sample actions for every agent
            self.action = [space.sample() if hasattr(space, 'sample') 
                          else random.randint(0, 7) for space in self.action_space]
        else:
            # common gym space obj
            self.action = self.action_space.sample()

    def act(self, keep_steps=10):
        self.count_steps += 1
        if self.count_steps > keep_steps:
            # sample new actions
            if isinstance(self.action_space, list):
                self.action = [space.sample() if hasattr(space, 'sample') 
                              else random.randint(0, 7) for space in self.action_space]
            else:
                self.action = self.action_space.sample()
            self.count_steps = 0
        
        return self.action

    def reset(self):
        # same logic
        if isinstance(self.action_space, list):
            self.action = [space.sample() if hasattr(space, 'sample') 
                          else random.randint(0, 7) for space in self.action_space]
        else:
            self.action = self.action_space.sample()
        self.count_steps = 0


import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time

class ReachablePointsCollector:
    def __init__(self, env, num_agents=5, grid_resolution=100.0):
        self.env = env
        self.num_agents = num_agents
        self.resolution = grid_resolution
        
        # Store reachable points in grid form to reduce memory usage and speed up duplicate detection
        self.reachable_grid = {}  # Keys are grid coordinates, values are actual coordinates
        self.agent_paths = [[] for _ in range(num_agents)]
        
        # Adjust number of agents in the environment
        self.env = augmentation.RandomPopulationWrapper(env, num_agents, num_agents, random_target=False)
        self.env = configUE.ConfigUEWrapper(self.env, offscreen=True)
        
        self.env.reset()
        # Create random agent
        self.agents = RandomAgent(self.env.action_space)

        # action clip
        self.action_clip = {}
        for i,obj in enumerate(self.env.unwrapped.player_list):
            self.action_clip[obj] = [1 for _ in range(8)]
            for j in range(8):
                clip_flag = random.randint(0, 1)
                if clip_flag == 1:
                    self.action_clip[obj][j] = 0.5
        # self.env.unwrapped.action_clip = self.action_clip
                    

    
    def _discretize_position(self, position):
        """Discretize continuous positions into grid coordinates"""
        x, y, z = position
        grid_x = round(x / self.resolution)
        grid_y = round(y / self.resolution)
        grid_z = round(z / self.resolution)
        return (grid_x, grid_y, grid_z)
    
    def collect(self,env_name, steps=5000, render=True, save_interval=500):
        """Start collecting reachable points
        
        Args:
            steps: Total steps
            render: Whether to render
            save_interval: Save intermediate results every this many steps
        """
        
        # Record initial positions
        initial_poses = self.get_all_agent_poses()
        for i, pose in enumerate(initial_poses):
            grid_pos = self._discretize_position(pose)
            self.reachable_grid[grid_pos] = pose
            self.agent_paths[i].append(pose)
        
        # Main collection loop
        
        pbar = tqdm.tqdm(range(steps), desc="Collecting reachable points", 
                     unit="steps", ncols=100, 
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    
        for step in pbar:
            # Execute random actions
            action = self.agents.act(keep_steps=1)
            _, _, _, _, _ = self.env.step(action)
            
            # Collect position information
            poses = self.get_all_agent_poses()
            for i, pose in enumerate(poses):
                grid_pos = self._discretize_position(pose)
                self.reachable_grid[grid_pos] = pose
                self.agent_paths[i].append(pose)

            # allow for rendering
            time.sleep(1.0 / 30)  # Assuming 30 FPS, adjust as needed

            # Display progress and visualization
            if step % 100 == 0:
                print(f"Step {step}/{steps}, collected {len(self.reachable_grid)} reachable points")
                if render:
                    self.visualize(save=True, filename=f"coverage_map_{step}.png")
            
            # Periodically save results
            if step % save_interval == 0 and step > 0:
                path = f"E:/EQA/unrealzoo_gym/example/reachable_point/{env_name}"
                if not os.path.exists(path):
                    os.makedirs(path)
                print(f"Saving reachable points at step {step} to {path}")
                self.save(f"E:/EQA/unrealzoo_gym/example/reachable_point/{env_name}/{env_name}_{step}.pkl")
        
        # Final save
        self.save("reachable_points_final.pkl")
        return self.reachable_grid
    
    def get_all_agent_poses(self):
        """Get positions of all agents"""
        poses = []
        env_unwrapped = self.env.unwrapped
        
        # Get positions of all agents from environment
        for i,obj in enumerate(env_unwrapped.player_list):
            # Get position of each agent
            try:
                pose = env_unwrapped.unrealcv.get_obj_location(obj)
            except:
                raise ValueError("API ERROR")     
            poses.append(pose)
        return poses
    
    def visualize(self, save=True, filename="coverage_map.png"):
        """Visualize collected reachable points"""
        # Extract x-y coordinates for 2D plot
        points = np.array(list(self.reachable_grid.values()))
        
        plt.figure(figsize=(10, 10))
        # Plot reachable points scatter
        plt.scatter(points[:, 0], points[:, 1], c='blue', s=1, alpha=0.5)
        
        # Plot paths of each agent
        colors = ['r', 'g', 'm', 'y', 'c', 'b', 'k', 'orange']
        for i, path in enumerate(self.agent_paths):
            if path:
                path_array = np.array(path)
                plt.plot(path_array[:, 0], path_array[:, 1], 
                         color=colors[i % len(colors)], 
                         linewidth=0.5, alpha=0.3,
                         label=f"Agent {i}")
        
        plt.title(f"Collected {len(self.reachable_grid)} reachable points")
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.legend()
        
        if save:
            path = f"E:/EQA/unrealzoo_gym/example/reachable_point/{env_name}/visualization"
            if not os.path.exists(path):
                os.makedirs(path)
            file_path = os.path.join(path, filename)
            plt.savefig(file_path, dpi=300)
            print(f"Map saved as {file_path}")
        else:
            plt.title(f"Collected {len(self.reachable_grid)} reachable points - Press any key to continue")
            plt.tight_layout()
            plt.draw()
            plt.waitforbuttonpress()
    
    def save(self, filename="reachable_points.pkl"):
        """Save collected points"""
        data = {
            "reachable_points": self.reachable_grid,
            "agent_paths": self.agent_paths,
            "resolution": self.resolution
        }
        path = f"E:/EQA/unrealzoo_gym/example/reachable_point/{env_name}"
        if not os.path.exists(path):
            os.makedirs(path)
        filename = os.path.join(path, filename)
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {len(self.reachable_grid)} reachable points to {filename}")
    
    @classmethod
    def load(cls, filename):
        """Load previously saved reachable points data"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        collector = cls(None, num_agents=0)  # Create empty collector
        collector.reachable_grid = data["reachable_points"]
        collector.agent_paths = data["agent_paths"]
        collector.resolution = data["resolution"]
        return collector




if __name__ == '__main__':
    # 定义要采集的环境列表
    env_list = [
    # "Map_ChemicalPlant_1",
    # "ModularNeighborhood",
    # "ModularSciFiVillage",
    # "RuralAustralia_Example_01",
    # "ModularVictorianCity",
    # "Cabin_Lake",
    # "Pyramid",
    "ModularGothic_Day",
    # "Greek_Island"
    ]
    
    # 遍历每个环境进行采集
    try:
        for env_name in env_list:
            print(f"\n========== 开始采集环境: {env_name} ==========\n")
            
            parser = argparse.ArgumentParser(description=None)
            parser.add_argument("-e", "--env_id", nargs='?', default=f'UnrealCv_Random_base-{env_name}-DiscreteRgbd-v0',
                                help='Select the environment to run')
            parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
            parser.add_argument("-s", '--seed', dest='seed', default=42, help='random seed')
            parser.add_argument("-t", '--time-dilation', dest='time_dilation', default=-1, help='time_dilation to keep fps in simulator')
            parser.add_argument("-n", '--nav-agent', dest='nav_agent', action='store_true', help='use nav agent to control the agents')
            parser.add_argument("-d", '--early-done', dest='early_done', default=-1, help='early_done when lost in n steps')
            parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')

            args = parser.parse_args([])  # 使用空列表避免从命令行解析参数
            
            # 为每个环境设置env_id
            args.env_id = f'UnrealCv_Random_base-{env_name}-DiscreteRgbd-v0'
            
            env = gym.make(args.env_id)
            if int(args.time_dilation) > 0:
                env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
            if int(args.early_done) > 0:
                env = early_done.EarlyDoneWrapper(env, int(args.early_done))
            if args.monitor:
                env = monitor.DisplayWrapper(env)
            agent_num = len(env.unwrapped.safe_start)
            # 为每个环境创建采集器
            collector = ReachablePointsCollector(env, num_agents=agent_num, grid_resolution=100)
            try:
                print(f"开始采集环境 {env_name} 的可达点...")
                collector.collect(env_name=env_name, steps=10000, render=True, save_interval=100)
                collector.visualize(save=True, filename=f"coverage_map_{env_name}.png")
                collector.save(f"{env_name}_reachable_points.pkl")
                env.close()
                print(f"\n========== 完成环境 {env_name} 的采集 ==========\n")
            except KeyboardInterrupt:
                print(f'用户中断，保存当前环境 {env_name} 的可达点')
                collector.save(f"{env_name}_reachable_points_interrupted.pkl")
                env.close()
                break  # 中断后退出整个循环
            except Exception as e:
                print(f"采集环境 {env_name} 时发生错误: {e}")
                env.close()
            finally:
                env.close()
    except KeyboardInterrupt:
        print("用户中断，停止采集所有环境")
        env.close()
    except Exception as e:
        print(f"发生错误: {e}") 
        env.close()
    finally:
        env.close()
    print("所有环境采集完成!")