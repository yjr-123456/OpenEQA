"""
Agent采样模块
用于生成各种类型的agents及其属性
"""

import random
import json
from collections import Counter
from typing import Dict, List, Any, Optional, Union
import string
import os
AGENT_OPTIONS = {
    "player": {
        "app_id": [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16, 17, 19],
        "animation": ["stand", "crouch", "liedown", "pick_up", "in_vehicle"]
    },
    "animal": {
        "app_id": [0, 1, 2, 3, 6, 10, 11, 12, 14, 15, 16, 19, 20, 21, 22, 23, 25, 26, 27],
        "animation": ["None"]
    },
    "drone": {
        "app_id": [0],
        "animation": ["none"]
    },
    "car": {
        "app_id": [0,1,2,3],
        "animation": ["None"],
        "type": ["BP_Hatchback_child_base_C","BP_Hatchback_child_extras_C","BP_Sedan_child_base_C", "BP_Sedan_child_extras_C"]
    },
    "motorbike": {
        "app_id": [0],
        "animation": ["None"]
    }
}

# ID-名称映射字典
AGENT_FEATURE_CAPTION = {
    "player": {
        1: "the_man_in_light_pants_and_sunglasses",
        2: "the_man_in_green_sweater",
        3: "the_woman_in_deep_V_top",
        4: "the_woman_in_floral_shirt",
        5: "the_woman_in_camo_vest",
        6: "the_woman_in_black_vest",
        7: "the_bald_man_in_glasses",
        8: "the_bald_man_in_burgundy_sweaters_and_grey_suit",
        9: "the_man_in_blue_polo",
        10: "the_man_in_burgundy_polo_and_light_pants",
        11: "the_man_in_white_shirt",
        12: "the_man_in_grey_suit",
        13: "the_man_in_blue_vest",
        15: "the_bald_man_in_brown_sweater_and_black_pants",
        16: "the_man_in_fedora_hat",
        17: "the_man_in_blue_plaid_pants_and_light_brown_turtleneck_sweater",
        19: "the_man_in_black_turleneck_sweater"
    },
    "animal": {
        0: "Beagle_Dog",
        1: "Great_Dane", 
        2: "Doberman_Pinscher",
        3: "Tabby_Cat",
        6: "Water_Buffalo",  #
        10: "Komodo_Dragon", #
        11: "Pig",             #
        12: "Spider",       
        14: "Camel",
        15: "Horse",
        16: "Puma",
        19: "Penguin",
        20: "Rhinoceros",
        21: "Tiger",
        22: "Zebra",
        23: "Elephant",
        25: "Turtle",
        26: "Snapping_Turtles",
        27: "Toucan"
    },
    "drone": {
        0: "White_Drone",
    },
    "car": {
        "BP_Hatchback_child_base_C": "Grey car, compact, four-door, SUV style, modern structure",
        "BP_Hatchback_child_extras_C": "White compact car with black grille, sleek design, and four doors",
        "BP_Sedan_child_base_C": "Orange car, sedan style, sleek shape, visible headlights and taillights",
        "BP_Sedan_child_extras_C": "Yellow vehicle with black stripes, modern style, compact structure",
        0: "Grey_SUV",
        1: "White_Compact_Car",
        2: "Orange_Sedan_Car",
        3: "Yellow_Car",
    },
    "motorbike": {
        0: "Motorbike"
    }
}

# 默认分布配置
DEFAULT_DISTRIBUTION = {
    "player": 0.3,
    "animal": 0.2,
    "car": 0.25,
    "drone": 0.15,
    "motorbike": 0.1
}

def load_json_file(file_path):
    """加载JSON文件"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"JSON解码错误: {file_path}")
        return {}

class AgentSampler:
    """Agent采样器类"""
    
    def __init__(self, custom_options=None, custom_distribution=None, feature_path = "./agent_caption/agent_feature_captions.json"):
        """
        初始化采样器
        
        Args:
            custom_options: 自定义agent选项，如果为None则使用默认选项
            custom_distribution: 自定义默认分布，如果为None则使用默认分布
            name_mapping: 自定义ID-名称映射，如果为None则使用默认映射
        """
        self.agent_options = custom_options or AGENT_OPTIONS
        self.default_distribution = custom_distribution or DEFAULT_DISTRIBUTION
        # self.feature_caption = feature_caption or AGENT_FEATURE_CAPTION
        current_path = os.path.dirname(os.path.abspath(__file__))
        feature_path = os.path.join(current_path, feature_path)
        self.feature_caption = load_json_file(feature_path)
        # print(f"feature_caption: {self.feature_caption}")
        self.batch_id_counter = 0

    def get_agent_caption(self, agent_type, app_id):
        str_id = str(app_id)
        return self.feature_caption.get(agent_type, {}).get(str_id, f"unknown_{agent_type}_{app_id}")
    # def get_agent_name(self, agent_type, appid, batch_id):
    #     return f"{agent_type}_{appid}_{batch_id}"  
    
    def get_agent_name(self, agent_type, appid_or_type, batch_id):
        # if agent_type == "car" and isinstance(appid_or_type, str) and appid_or_type in self.agent_options["car"].get("type", []):
        #     # 如果是car类型且提供的是type字符串，直接使用type作为名称一部分
        #     return f"{appid_or_type}_{batch_id}"
        # else:
            # 常规情况下使用app_id
        return f"{agent_type}_{appid_or_type}_{batch_id}"

    @staticmethod
    def _partition_number(total_sum: int, num_parts: int) -> List[int]:
        if num_parts <= 0:
            return []
        if total_sum < num_parts:
            return []
        if num_parts == 1:
            return [total_sum]
        try:
            split_points = sorted(random.sample(range(1, total_sum), num_parts - 1))
        except ValueError:
            return []
            
        parts = []
        last_split = 0
        for point in split_points:
            parts.append(point - last_split)
            last_split = point
        parts.append(total_sum - last_split)
        return parts

    def _generate_batch_id(self):
        batch_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        self.batch_id_counter += 1
        return batch_id

    def sample_agents(self, total_agents=10, agent_distribution=None):
        """
        采样指定数量的agents
        
        Args:
            total_agents: 要采样的agent总数
            agent_distribution: 指定各类agent比例，例如 {'player': 0.3, 'car': 0.2}
        
        Returns:
            dict: 按类别组织的agent配置
        """
        # 使用默认分布如果未指定
        if agent_distribution is None:
            agent_distribution = self.default_distribution
        
        # 确保分布仅包含可用类型
        valid_distribution = {k: v for k, v in agent_distribution.items() 
                            if k in self.agent_options}
        
        # 重新归一化分布
        total = sum(valid_distribution.values())
        if total > 0:
            valid_distribution = {k: v/total for k, v in valid_distribution.items()}
        else:
            return {}
        
        # 计算各类型agent的数量
        agent_counts = {}
        remaining = total_agents
        
        agent_types = list(valid_distribution.keys())
        for i, agent_type in enumerate(agent_types[:-1]):
            count = int(total_agents * valid_distribution[agent_type])
            agent_counts[agent_type] = count
            remaining -= count
        
        # 剩余的分配给最后一项
        if agent_types:
            agent_counts[agent_types[-1]] = max(0, remaining)
        
        # 使用具体数量采样
        return self.sample_with_specific_counts(agent_counts)
        
    # def sample_with_specific_counts_no_repeat(self, agent_counts):
    #     """
    #     根据具体数量采样agents，尽量避免重复
    #     """
    #     self.batch_id = self._generate_batch_id()  # 为这批采样生成一个批次ID
    #     sampled_agents = {}

    #     has_vehicles = any(agent_type in ["car", "motorbike"] for agent_type in agent_counts.keys())
        
    #     for agent_type, count in agent_counts.items():
    #         if count <= 0 or agent_type not in self.agent_options:
    #             continue
                
    #         options = self.agent_options[agent_type]
            
    #         available_animations = options["animation"].copy()
    #         car_count = agent_counts.get("car", 0)
    #         motorbike_count = agent_counts.get("motorbike", 0)
    #         if agent_type == "player":
    #             if (not has_vehicles or (car_count + motorbike_count == 0)) and "in_vehicle" in available_animations:
    #                 available_animations.remove("in_vehicle")
    #                 print(f"No vehicles present, removing 'in_vehicle' animation for player")
    #         # 准备结果字段
    #         sampled_app_ids = []
    #         sampled_animations = []
    #         sampled_types = []
            
    #         # 使用random.sample采样app_id (如果数量足够)
    #         available_app_ids = options["app_id"]
    #         if count <= len(available_app_ids):
    #             # 不重复抽取app_id
    #             sampled_app_ids = random.sample(available_app_ids, count)
    #         else:
    #             # 数量不够时允许重复
    #             sampled_app_ids = [random.choice(available_app_ids) for _ in range(count)]
            
    #         # 同样对animation应用random.sample逻辑
    #         available_animations = available_animations
    #         if count <= len(available_animations):
    #             sampled_animations = random.sample(available_animations, count)
    #         else:
    #             sampled_animations = [random.choice(available_animations) for _ in range(count)]
            
    #         # 生成agent名称
    #         names = [self.get_agent_name(agent_type, appid, self.batch_id) for appid in sampled_app_ids]
    #         # 生成agent特征描述
    #         captions = [self.get_agent_caption(agent_type, app_id) for app_id in sampled_app_ids]
            
    #         # 保存到结果
    #         sampled_agents[agent_type] = {
    #             "name": names,
    #             "app_id": sampled_app_ids,
    #             "animation": sampled_animations,
    #             "feature_caption": captions  # 添加特征描述字段
    #         }
        
    #     return sampled_agents, AGENT_FEATURE_CAPTION
    def sample_with_specific_counts_no_repeat(self, agent_counts):
        """
        根据具体数量采样agents，尽量避免重复
        """
        self.batch_id = self._generate_batch_id()  # 为这批采样生成一个批次ID
        sampled_agents = {}

        has_vehicles = any(agent_type in ["car", "motorbike"] for agent_type in agent_counts.keys())
        
        for agent_type, count in agent_counts.items():
            if count <= 0 or agent_type not in self.agent_options:
                continue
                
            options = self.agent_options[agent_type]
            
            available_animations = options["animation"].copy()
            car_count = agent_counts.get("car", 0)
            motorbike_count = agent_counts.get("motorbike", 0)
            if agent_type == "player":
                if (not has_vehicles or (car_count + motorbike_count == 0)) and "in_vehicle" in available_animations:
                    available_animations.remove("in_vehicle")
                    print(f"No vehicles present, removing 'in_vehicle' animation for player")
            
            # 准备结果字段
            sampled_app_ids = []
            sampled_animations = []
            sampled_types = []  # 新增：用于存储car类型
            
            # 特殊处理car类型，使用type字段而非app_id
            if agent_type == "car" and "type" in options:
                # 使用type字段进行采样
                available_ids = options["app_id"]
                if count <= len(available_ids):
                    # 不重复抽取car类型
                    sampled_app_ids = random.sample(available_ids, count)
                else:
                    # 数量不够时允许重复
                    sampled_app_ids = [random.choice(available_ids) for _ in range(count)]
                
                # 对于car类型，app_id保持默认值(0)
                sampled_types= [options["type"][sampled_app_ids[i]] for i in range(count)]
            else:
                # 其他类型正常处理app_id
                available_app_ids = options["app_id"]
                if count <= len(available_app_ids):
                    # 不重复抽取app_id
                    sampled_app_ids = random.sample(available_app_ids, count)
                else:
                    # 数量不够时允许重复
                    sampled_app_ids = [random.choice(available_app_ids) for _ in range(count)]
            
            # 采样动画
            if count <= len(available_animations):
                sampled_animations = random.sample(available_animations, count)
            else:
                sampled_animations = [random.choice(available_animations) for _ in range(count)]
            
            # 生成agent名称
            if agent_type == "car":
                # 对于car类型，使用type作为名称的一部分
                names = [self.get_agent_name(agent_type, appid, self.batch_id) for appid in sampled_app_ids]
                captions = [self.get_agent_caption(agent_type, sampled_agent) for sampled_agent in sampled_types]
            else:
                names = [self.get_agent_name(agent_type, appid, self.batch_id) for appid in sampled_app_ids]
                captions = [self.get_agent_caption(agent_type, app_id) for app_id in sampled_app_ids]
            # 保存到结果
            result = {
                "name": names,
                "app_id": sampled_app_ids,
                "animation": sampled_animations,
                "feature_caption": captions
            }
            
            # 如果是car类型，添加type字段
            if agent_type == "car":
                result["type"] = sampled_types
                
            sampled_agents[agent_type] = result
        
        return sampled_agents, AGENT_FEATURE_CAPTION
    

    def add_names_to_output(self, sampled_agents):
        """为输出结果添加名称和特征描述"""
        result = {}
        for agent_type, config in sampled_agents.items():
            app_ids = config["app_id"]

            # 基于类型和索引生成名称
            names = [self.get_agent_name(agent_type, app_id, self.batch_id) for app_id in app_ids]
            captions = [self.get_agent_caption(agent_type, app_id) for app_id in app_ids]
            
            # 创建包含名称和特征描述的新配置
            result[agent_type] = {
                **config,
                "name": names,  # 添加名称字段
                "feature_caption": captions  # 添加特征描述字段
            }
        return result
    
    def save_to_json(self, sampled_agents, output_file, include_names=True):
        """
        将采样结果保存为JSON文件
        
        Args:
            sampled_agents: 采样结果
            output_file: 输出文件路径
            include_names: 是否在输出中包含名称
        """
        # 决定是否添加名称
        output_data = self.add_names_to_output(sampled_agents) if include_names else sampled_agents
        
        with open(output_file, 'w') as f:
            json.dump({"target_configs": output_data}, f, indent=4)
        print(f"已保存到 {output_file}")
    
    def print_stats(self, sampled_agents):
        """打印采样统计信息"""
        print("采样完成:")
        total_agents = 0
        for agent_type, config in sampled_agents.items():
            count = len(config['app_id'])
            total_agents += count
            print(f"  - {agent_type}: {count} 个")
            
            # 详细分布
            app_id_counter = Counter(config['app_id'])
            print(f"    app_id分布: {dict(app_id_counter)}")
            
            # 显示对应的名称
            print(f"    对应名称: ", end="")
            for i in range(count):
                name = self.get_agent_name(agent_type, i+1,self.batch_id)
                print(f"{name}, ", end="")
            print()
            
            # 动画分布
            animation_counter = Counter(config['animation'])
            print(f"    动作分布: {dict(animation_counter)}")
        
        print(f"总计: {total_agents} 个agents")

    # def sample_identical_agents(self, agent_type: str, count: int):
    #     if count <= 0 or agent_type not in self.agent_options:
    #         return {}

    #     options = self.agent_options[agent_type]
        
    #     if count > 2 and len(options["app_id"]) > 2:
    #         obj_num = random.randint(1, 3)
    #         app_ids = random.sample(options["app_id"], obj_num)
    #         count_list = self._partition_number(count, obj_num)
    #         print(count, app_ids, count_list)
    #         assert len(app_ids) == len(count_list)
    #     else:
    #         count_list = [count]
    #         app_ids = random.choices(options["app_id"])
    
    #     self.batch_id = self._generate_batch_id()

    #     sampled_agents = {}

    #     for i, app_id in enumerate(app_ids):
    #         sampled_app_ids = [app_id] * count_list[i]
    #         sampled_animations = [random.choice(options["animation"]) for _ in range(count_list[i])]
        
    #         names = [self.get_agent_name(agent_type, app_id ,f"{self.batch_id}_{i+1}") for i in range(count_list[i])]
    #         captions = [self.get_agent_caption(agent_type, app_id)] * count_list[i]
    #         if i == 0:
    #             sampled_agents = {
    #                 agent_type: {
    #                     "name": names,
    #                     "app_id": sampled_app_ids,
    #                     "animation": sampled_animations,
    #                     "feature_caption": captions
    #                 }
    #             }
    #         else:
    #             exist_data = sampled_agents.get(agent_type, {})
    #             exist_data["name"].extend(names)
    #             exist_data["app_id"].extend(sampled_app_ids)
    #             exist_data["animation"].extend(sampled_animations)
    #             exist_data["feature_caption"].extend(captions)
    #             sampled_agents[agent_type] = exist_data
    #     return sampled_agents
    def sample_identical_agents(self, agent_type: str, count: int):
        if count <= 0 or agent_type not in self.agent_options:
            return {}
    
        options = self.agent_options[agent_type]
        self.batch_id = self._generate_batch_id()
        sampled_agents = {}
        
        # 特殊处理car类型
        if agent_type == "car" and "type" in options:
            # 使用car类型进行采样
            if count > 2 and len(options["type"]) > 2:
                obj_num = random.randint(1, 3)
                car_types = random.sample(options["type"], obj_num)
                count_list = self._partition_number(count, obj_num)
                print(f"采样car类型: {count}辆, 选择了{obj_num}种车型: {car_types}, 分布: {count_list}")
            else:
                count_list = [count]
                car_types = [random.choice(options["type"])]
                
            # 生成car信息
            for i, car_type in enumerate(car_types):
                sampled_app_ids = [options["app_id"][0]] * count_list[i]  # 使用默认app_id
                sampled_types = [car_type] * count_list[i]
                sampled_animations = [random.choice(options["animation"]) for _ in range(count_list[i])]
                
                # 使用car类型作为名称的一部分
                names = [f"{car_type}_{self.batch_id}_{j}" for j in range(count_list[i])]
                captions = [self.get_agent_caption(agent_type, options["app_id"][0])] * count_list[i]
                
                if i == 0:
                    sampled_agents = {
                        agent_type: {
                            "name": names,
                            "app_id": sampled_app_ids,
                            "type": sampled_types,
                            "animation": sampled_animations,
                            "feature_caption": captions
                        }
                    }
                else:
                    exist_data = sampled_agents.get(agent_type, {})
                    exist_data["name"].extend(names)
                    exist_data["app_id"].extend(sampled_app_ids)
                    exist_data["type"].extend(sampled_types)
                    exist_data["animation"].extend(sampled_animations)
                    exist_data["feature_caption"].extend(captions)
                    sampled_agents[agent_type] = exist_data
        else:
            # 非car类型的原始处理逻辑
            if count > 2 and len(options["app_id"]) > 2:
                obj_num = random.randint(1, 3)
                app_ids = random.sample(options["app_id"], obj_num)
                count_list = self._partition_number(count, obj_num)
                print(count, app_ids, count_list)
                assert len(app_ids) == len(count_list)
            else:
                count_list = [count]
                app_ids = random.choices(options["app_id"])
    
            for i, app_id in enumerate(app_ids):
                sampled_app_ids = [app_id] * count_list[i]
                sampled_animations = [random.choice(options["animation"]) for _ in range(count_list[i])]
            
                names = [self.get_agent_name(agent_type, app_id ,f"{self.batch_id}_{i+1}") for i in range(count_list[i])]
                captions = [self.get_agent_caption(agent_type, app_id)] * count_list[i]
                if i == 0:
                    sampled_agents = {
                        agent_type: {
                            "name": names,
                            "app_id": sampled_app_ids,
                            "animation": sampled_animations,
                            "feature_caption": captions
                        }
                    }
                else:
                    exist_data = sampled_agents.get(agent_type, {})
                    exist_data["name"].extend(names)
                    exist_data["app_id"].extend(sampled_app_ids)
                    exist_data["animation"].extend(sampled_animations)
                    exist_data["feature_caption"].extend(captions)
                    sampled_agents[agent_type] = exist_data
        
        return sampled_agents

    def sample_agent_typid(self, agent_type_category: List[str], agent_num:int):
        count_list = self._partition_number(agent_num, len(agent_type_category))
        sampled_agents_final = {}
        for i,agent_type in enumerate(agent_type_category):
            count = count_list[i]
            sampled_agents = self.sample_identical_agents(agent_type, count)
            sample_data = sampled_agents.get(agent_type, {})
            if agent_type not in sampled_agents_final:
                sampled_agents_final[agent_type] = sample_data
            else:
                exist_data = sampled_agents_final[agent_type]
                for key in ["name", "app_id", "animation", "feature_caption"]:
                    if key in sample_data:
                        exist_data[key].extend(sample_data[key])
                sampled_agents_final[agent_type] = exist_data
        return sampled_agents_final, AGENT_FEATURE_CAPTION



if __name__ == "__main__":
    sampler = AgentSampler()
    agent_num = 5
    agent_type = ["player", "animal", "car", "motorbike"]
    agent_counts = {
        "player": 5,
        "animal": 1,
        "car": 3,
        "motorbike": 0
    }
    x,y = sampler.sample_with_specific_counts_no_repeat(agent_counts)
    print(x)
    print(y)
       #print(sampler.sample_agent_typid(agent_type_category=agent_type, agent_num=agent_num))

