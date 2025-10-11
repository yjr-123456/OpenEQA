import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import argparse
#import gym_unrealcv
import gymnasium as gym
# from gymnasium import wrappers
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
import math
import  matplotlib
matplotlib.font_manager.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc")
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
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

class RegionConstrainedRandomAgent(object):
    """限制在特定区域内移动的随机智能体"""
    def __init__(self, action_space, env, sample_region=None):
        self.action_space = action_space
        self.env = env
        self.count_steps = 0
        self.sample_region = sample_region  # 格式: [[x_min, x_max], [y_min, y_max]]
        
        # 获取动作定义
        try:
            self.move_actions = self.env.unwrapped.player_move_action
            print(f"已加载动作定义: {len(self.move_actions)} 个动作")
        except:
            # 如果获取失败，使用默认动作定义（角度,距离）
            self.move_actions = [
                [0, 100],     # 前进
                [0, -100],    # 后退
                [15, 50],     # 右前
                [-15, 50],    # 左前
                [30, 0],      # 右转
                [-30, 0],     # 左转
                [0, 0]        # 停止
            ]
            print("使用默认动作定义")
        
        # 初始化动作
        if isinstance(self.action_space, list):
            self.action = [self._sample_valid_action(i) for i in range(len(self.action_space))]
        else:
            self.action = self.action_space.sample()

    def _predict_next_position(self, current_pos, current_rot, action_idx):
        """预测执行某个动作后的新位置
        
        Args:
            current_pos: 当前位置 [x, y, z]
            current_rot: 当前旋转 [pitch, yaw, roll]
            action_idx: 动作索引
            
        Returns:
            预测的下一个位置 [x, y, z]
        """
        # 防止无效的动作索引
        if action_idx >= len(self.move_actions):
            return current_pos
            
        # 获取动作定义（角度变化,距离）
        action = self.move_actions[action_idx]
        
        # 根据动作类型决定如何计算
        if len(action) == 2:  # 标准移动动作 [角度, 距离]
            angle_change, distance = action
            
            # 当前朝向（UE是左手系，角度增加表示向右旋转）
            current_angle = current_rot[1]  # yaw角度
            
            # 计算新的朝向角度（弧度）
            new_angle_rad = math.radians(current_angle + angle_change)
            
            # 计算位移                                                                                                                                                                                                                                                                                                                                                                                           
            dx = distance * math.cos(new_angle_rad)
            dy = distance * math.sin(new_angle_rad)
            
            # 预测的新位置
            new_x = current_pos[0] + dx
            new_y = current_pos[1] + dy
            new_z = current_pos[2]  # 保持z坐标不变
            
            return [new_x, new_y, new_z]
            
        elif len(action) == 4:  # 无人机动作 [前后, 左右, 上下, 旋转]
            # 简化的无人机动作预测
            forward, right, up, _ = action
            
            # 当前朝向（弧度）
            current_angle_rad = math.radians(current_rot[1])
            
            # 计算全局坐标系中的位移
            dx = forward * 100 * math.cos(current_angle_rad) - right * 100 * math.sin(current_angle_rad)
            dy = forward * 100 * math.sin(current_angle_rad) + right * 100 * math.cos(current_angle_rad)
            dz = up * 100
            
            # 预测的新位置
            new_x = current_pos[0] + dx
            new_y = current_pos[1] + dy
            new_z = current_pos[2] + dz
            
            return [new_x, new_y, new_z]
            
        else:
            # 未知动作格式
            return current_pos
    
    def _is_in_region(self, position):
        """检查位置是否在采样区域内"""
        if self.sample_region is None:
            return True
            
        x, y = position[0], position[1]  # 只检查x和y坐标
        x_range, y_range = self.sample_region
        
        return (x_range[0] <= x <= x_range[1]) and (y_range[0] <= y <= y_range[1])
    
    def _get_agent_pose(self, agent_idx):
        """获取指定智能体的位置和旋转"""
        try:
            agent_name = self.env.unwrapped.player_list[agent_idx]
            location = self.env.unwrapped.unrealcv.get_obj_location(agent_name)
            rotation = self.env.unwrapped.unrealcv.get_obj_rotation(agent_name)
            return location, rotation
        except Exception as e:
            print(f"获取智能体位置失败: {e}")
            return [0, 0, 0], [0, 0, 0]  # 默认值
    
    def _sample_valid_action(self, agent_idx):
        """采样一个不会导致智能体离开区域的有效动作"""
        if self.sample_region is None:
            # 如果没有区域限制，直接返回随机动作
            return random.randint(0, len(self.move_actions) - 1)
            
        # 获取智能体当前位置和旋转
        current_pos, current_rot = self._get_agent_pose(agent_idx)
            
        # 如果当前已经在区域外，选择可能使其返回区域的动作
        if not self._is_in_region(current_pos):
            # 计算到区域中心的方向
            x_center = (self.sample_region[0][0] + self.sample_region[0][1]) / 2
            y_center = (self.sample_region[1][0] + self.sample_region[1][1]) / 2
            
            # 计算从当前位置到区域中心的向量
            to_center_x = x_center - current_pos[0]
            to_center_y = y_center - current_pos[1]
            
            # 计算目标角度（弧度）
            target_angle_rad = math.atan2(to_center_y, to_center_x)
            
            # 将目标角度转换为度数
            target_angle_deg = math.degrees(target_angle_rad)
            
            # 计算当前朝向与目标方向的角度差
            current_angle = current_rot[1]  # yaw角度
            angle_diff = (target_angle_deg - current_angle + 180) % 360 - 180
            
            # 选择最可能返回区域的动作
            if abs(angle_diff) < 30:
                # 朝向大致正确，向前移动
                forward_actions = [i for i, a in enumerate(self.move_actions) if len(a) == 2 and a[1] > 0]
                if forward_actions:
                    return random.choice(forward_actions)
            elif angle_diff > 0:
                # 需要向右转
                right_turn_actions = [i for i, a in enumerate(self.move_actions) if len(a) == 2 and a[0] > 0]
                if right_turn_actions:
                    return random.choice(right_turn_actions)
            else:
                # 需要向左转
                left_turn_actions = [i for i, a in enumerate(self.move_actions) if len(a) == 2 and a[0] < 0]
                if left_turn_actions:
                    return random.choice(left_turn_actions)
            
            # 如果没有找到特定动作，随机选择
            return random.randint(0, len(self.move_actions) - 1)
        
        # 尝试预测每个动作的结果，找到保持在区域内的动作
        valid_actions = []
        for action_idx in range(len(self.move_actions)):
            next_pos = self._predict_next_position(current_pos, current_rot, action_idx)
            if self._is_in_region(next_pos):
                valid_actions.append(action_idx)
        
        # 如果有有效动作，随机选择一个
        if valid_actions:
            return random.choice(valid_actions)
        else:
            # 如果没有有效动作，随机选择任意动作
            return random.randint(0, len(self.move_actions) - 1)
            
    def act(self, keep_steps=10):
        """产生动作，每keep_steps步重新采样一次"""
        self.count_steps += 1
        if self.count_steps > keep_steps:
            # 重新采样动作
            # if isinstance(self.action_space, tuple) or isinstance(self.action_space, list):
            #     self.action = [self._sample_valid_action(i) for i in range(len(self.action_space))]
            if len(self.action_space) > 1:
                # 多个智能体情况
                self.action = tuple([self._sample_valid_action(i) for i in range(len(self.action_space))])
            else:
                # 单个智能体情况
                self.action = self._sample_valid_action(0)
            self.count_steps = 0
        
        return self.action

    def reset(self):
        """重置智能体状态"""
        if isinstance(self.action_space, list):
            self.action = [self._sample_valid_action(i) for i in range(len(self.action_space))]
        else:
            self.action = self._sample_valid_action(0)
        self.count_steps = 0


class ReachablePointsCollector:
    def __init__(self, env, num_agents=5, grid_resolution=100.0, sample_region=None):
        self.env = env
        self.num_agents = num_agents
        self.resolution = grid_resolution
        self.sample_region = sample_region
        
        # 存储可达点
        self.reachable_grid = {}
        self.agent_paths = [[] for _ in range(num_agents)]
        
        # 环境包装
        self.env = augmentation.RandomPopulationWrapper(env, num_agents, num_agents, random_target=False)
        self.env = configUE.ConfigUEWrapper(self.env, offscreen=False)
        # self.move_action = self.env.unwrapped.move_action
        self.env.reset()
        
        # 使用改进的区域约束智能体
        if sample_region:
            print(f"启用区域约束: X范围={sample_region[0]}, Y范围={sample_region[1]}")
            self.agents = RegionConstrainedRandomAgent(self.env.action_space, self.env, sample_region)
        else:
            self.agents = RandomAgent(self.env.action_space)

        # action clip 不变
        self.action_clip = {}
        for i,obj in enumerate(self.env.unwrapped.player_list):
            self.action_clip[obj] = [1 for _ in range(8)]
            for j in range(8):
                clip_flag = random.randint(0, 1)
                if clip_flag == 1:
                    self.action_clip[obj][j] = 0.5

    def _discretize_position(self, position):
        """将连续位置离散化为网格坐标"""
        x, y, z = position
        grid_x = round(x / self.resolution)
        grid_y = round(y / self.resolution)
        grid_z = round(z / self.resolution)
        return (grid_x, grid_y, grid_z)
    
    def _is_in_sample_region(self, position):
        """检查位置是否在采样区域内"""
        if self.sample_region is None:
            return True  # 如果没有设定区域限制，总是返回True
            
        x, y, z = position
        x_range, y_range = self.sample_region
        
        # 检查x和y坐标是否在范围内
        return (x_range[0] <= x <= x_range[1]) and (y_range[0] <= y <= y_range[1])
    
    def get_all_agent_poses(self):
        """获取所有智能体的位置"""
        poses = []
        for agent_name in self.env.unwrapped.player_list:
            pos = self.env.unwrapped.unrealcv.get_obj_location(agent_name)
            poses.append(pos)
        return poses
    
    def collect(self, env_name, steps=5000, render=True, save_interval=500, save_path="."):
        """开始收集可达点
        
        Args:
            env_name: 环境名称，用于保存文件
            steps: 总步数
            render: 是否渲染
            save_interval: 每隔多少步保存一次中间结果
        """
        
        # 记录初始位置
        initial_poses = self.get_all_agent_poses()
        for i, pose in enumerate(initial_poses):
            # 只有在区域内的点才添加到可达点集合
            if self._is_in_sample_region(pose):
                grid_pos = self._discretize_position(pose)
                self.reachable_grid[grid_pos] = pose
                self.agent_paths[i].append(pose)
            else:
                self.agent_paths[i].append(pose)  # 仍然记录路径，但不添加到可达点
        
        # 判断是否有智能体在采样区域内
        agents_in_region = sum(1 for pose in initial_poses if self._is_in_sample_region(pose))
        if agents_in_region == 0 and self.sample_region is not None:
            print("警告：没有智能体在指定采样区域内。请尝试重置环境或调整采样区域。")
        
        # 主采集循环
        pbar = tqdm.tqdm(range(steps), desc="正在收集可达点", 
                     unit="步", ncols=100, 
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    
        for step in pbar:
            # 执行随机动作
            action = self.agents.act(keep_steps=1)
            _, _, _, _, _ = self.env.step(action)
            time.sleep(0.1)
            # 收集位置信息
            poses = self.get_all_agent_poses()
            for i, pose in enumerate(poses):
                # 记录智能体路径
                self.agent_paths[i].append(pose)
                
                # 只有在采样区域内的点才添加到可达点集合
                if self._is_in_sample_region(pose):
                    grid_pos = self._discretize_position(pose)
                    self.reachable_grid[grid_pos] = pose

            # 显示进度和可视化
            if step % 100 == 0:
                # 统计区域内的点数
                if self.sample_region is not None:
                    region_points = sum(1 for pos in self.reachable_grid.values() 
                                      if self._is_in_sample_region(pos))
                    print(f"步骤 {step}/{steps}, 采样区域内收集了 {region_points} 个可达点，总共 {len(self.reachable_grid)} 个")
                else:
                    print(f"步骤 {step}/{steps}, 收集了 {len(self.reachable_grid)} 个可达点")
                    
                if render:
                    self.visualize(save=True, env_name=env_name, save_path=save_path, filename=f"coverage_map_{step}.png")
            
            # 定期保存结果
            if step % save_interval == 0 and step > 0:
                path = f"{save_path}"
                if not os.path.exists(path):
                    os.makedirs(path)
                print(f"在步骤 {step} 保存可达点到 {path}")
                self.save(f"{path}/{env_name}_{step}.pkl")
        
        # 最终保存
        final_path = f"{save_path}/final"
        if not os.path.exists(final_path):
            os.makedirs(final_path)
        self.save(f"{final_path}/{env_name}_final.pkl")
        return self.reachable_grid

    def visualize(self, save=True, env_name="unknown", filename="coverage_map.png", save_path="."):
        """可视化收集的可达点"""
        # 提取x-y坐标用于2D绘图
        if not self.reachable_grid:
            print("警告: 没有可达点可以可视化")
            return
            
        points = np.array(list(self.reachable_grid.values()))
        
        plt.figure(figsize=(10, 10))
        
        # 如果有采样区域，绘制采样区域边界
        if self.sample_region is not None:
            x_range, y_range = self.sample_region
            x_min, x_max = x_range
            y_min, y_max = y_range
            
            # 绘制区域矩形
            rect = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, 
                               fill=False, edgecolor='red', linestyle='--', linewidth=2,
                               label='采样区域')
            plt.gca().add_patch(rect)
            
            # 筛选区域内的点
            region_points = np.array([pos for pos in self.reachable_grid.values() 
                                   if self._is_in_sample_region(pos)])
            
            if len(region_points) > 0:
                plt.scatter(region_points[:, 0], region_points[:, 1], 
                          c='green', s=2, alpha=0.7, label='区域内点')
            
            # 筛选区域外的点
            outside_points = np.array([pos for pos in self.reachable_grid.values() 
                                    if not self._is_in_sample_region(pos)])
            
            if len(outside_points) > 0:
                plt.scatter(outside_points[:, 0], outside_points[:, 1], 
                          c='blue', s=1, alpha=0.3, label='区域外点')
        else:
            # 原来的绘图逻辑
            plt.scatter(points[:, 0], points[:, 1], c='blue', s=1, alpha=0.5)
        
        # 绘制每个智能体的路径
        colors = ['r', 'g', 'm', 'y', 'c', 'b', 'k', 'orange']
        for i, path in enumerate(self.agent_paths):
            if path:
                path_array = np.array(path)
                plt.plot(path_array[:, 0], path_array[:, 1], 
                         color=colors[i % len(colors)], 
                         linewidth=0.5, alpha=0.3,
                         label=f"智能体 {i}")
        
        if self.sample_region:
            region_points_count = sum(1 for pos in self.reachable_grid.values() 
                                   if self._is_in_sample_region(pos))
            title = f"收集了 {region_points_count} 个区域内点 (总计: {len(self.reachable_grid)})"
        else:
            title = f"收集了 {len(self.reachable_grid)} 个可达点"
            
        plt.title(title)
        plt.xlabel("X 坐标")
        plt.ylabel("Y 坐标")
        plt.legend()
        
        if save:
            path = f"{save_path}/visualization"
            if not os.path.exists(path):
                os.makedirs(path)
            file_path = os.path.join(path, filename)
            plt.savefig(file_path, dpi=300)
            print(f"地图已保存为 {file_path}")
        else:
            plt.tight_layout()
            plt.draw()
            plt.waitforbuttonpress()
        
        plt.close()
    
    def save(self, filename):
        """保存可达点数据到文件"""
        data = {
            'reachable_points': self.reachable_grid,
            'agent_paths': self.agent_paths,
            'resolution': self.resolution,
            'sample_region': self.sample_region
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"数据已保存到 {filename}")


if __name__ == '__main__':
    # 定义要采集的环境列表
    # env_list = {
    # "Map_ChemicalPlant_1" : 5000
    # # "ModularGothic_Day": 3000,
    # # "Greek_Island": 2500,
   
    # # "Pyramid": 1500,
    # #  "Cabin_Lake": 4000,
    # # "ModularSciFiVillage": 2000,
    # # "RuralAustralia_Example_01": 5000,

    # # "ModularVictorianCity": 5000,
    # # "ModularNeighborhood": 5000,

    # }
    
    parser = argparse.ArgumentParser(description=None)
    # parser.add_argument("-e", "--env_id", nargs='?', default=f'UnrealCv_Random_base-{env_name}-DiscreteRgbd-v0',
    #                     help='Select the environment to run')
    parser.add_argument("-e", "--env_name", dest='env_name', default="Map_ChemicalPlant_1", help='environment name')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=42, help='random seed')
    parser.add_argument("-t", '--time-dilation', dest='time_dilation', default=-1, help='time_dilation to keep fps in simulator')
    parser.add_argument("-n", '--nav-agent', dest='nav_agent', action='store_true', help='use nav agent to control the agents')
    parser.add_argument("-d", '--early-done', dest='early_done', default=-1, help='early_done when lost in n steps')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')
    parser.add_argument("--max_step", dest='max_step', type=int, default=5000, help='max steps to collect reachable points')
    parser.add_argument("--save_path", dest='save_path', default=os.path.dirname(__file__), help='where to save the results')

    args = parser.parse_args([])  # 使用空列表避免从命令行解析参数
    env_name = args.env_name
    max_step = args.max_step
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
    sample_region = env.unwrapped.sample_region if hasattr(env.unwrapped, 'sample_region') else None
    collector = ReachablePointsCollector(env, num_agents=agent_num, grid_resolution=100, sample_region=sample_region)
    save_path = os.path.join(args.save_path, 'reachable_points', env_name)
    try:
        print(f"开始采集环境 {env_name} 的可达点...")
        collector.collect(env_name=env_name, steps=max_step, render=True, save_interval=500)
        collector.visualize(save=True, save_path=save_path, filename=f"coverage_map_{env_name}.png")
        collector.save(os.path.join(save_path, f"{env_name}_reachable_points.pkl"))
        env.close()
        print(f"\n========== 完成环境 {env_name} 的采集 ==========\n")
    except KeyboardInterrupt:
        print(f'用户中断，保存当前环境 {env_name} 的可达点')
        collector.save(os.path.join(save_path, f"{env_name}_reachable_points_interrupted.pkl"))
        env.close()
    except Exception as e:
        print(f"采集环境 {env_name} 时发生错误: {e}")
        env.close()
        collector.save(os.path.join(save_path, f"{env_name}_reachable_points_error.pkl"))
        import traceback
        traceback.print_exc()   
    finally:
        env.close()
    print("所有环境采集完成!")