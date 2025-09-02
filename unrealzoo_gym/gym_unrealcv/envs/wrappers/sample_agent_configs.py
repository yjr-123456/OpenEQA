from gymnasium import Wrapper
from example.agent_configs_sampler import AgentSampler, GraphBasedSampler
import random
from gym_unrealcv.envs.utils import misc
import time
from gym_unrealcv.envs.wrappers.augmentation import ConfigGenerator
class SampleAgentConfigWrapper(Wrapper):
    def __init__(self, env, agent_category, min_types=1, max_types=5, graph_path=None, 
                 type_count_ranges=None, min_total_agents=5, max_total_agents=10, if_cnt=False):  # 新增参数
        super().__init__(env)
        # set parameters
        self.agent_category = agent_category
        self.min_types = min_types
        self.max_types = max_types
        self.type_count_ranges = type_count_ranges  
        self.min_total_agents = min_total_agents    
        self.max_total_agents = max_total_agents
        self.if_cnt = if_cnt    
        # init sampler
        self.agent_sampler = AgentSampler()
        self.point_sampler = GraphBasedSampler(graph_path)

    def step(self,action):
        obs, reward, termination,truncation, info = self.env.step(action)
        return obs, reward, termination,truncation, info
    
    def reset(self, **kwargs):
        env = self.env.unwrapped
        
        if not env.launched:  # we need to launch the environment
            env.launched = env.launch_ue_env()
            env.init_agents()
            env.init_objects()
        # remove target agents
        if env.target_list != []:
            env.init_agents()

        vehicle_zones = env.unwrapped.vehicle_zones
        height_bias = env.unwrapped.height_bias
        # sample agent num for each agent type
        
        if self.if_cnt == False:
            agent_type_counts = self.sample_agent_types() 
            print("agent_type_counts\n",agent_type_counts)
            env.refer_agents_category = list(agent_type_counts.keys())        
            env.num_agents = sum(agent_type_counts.values()) + len(env.player_list)
            sampled_agent, name_mapping_dict = self.agent_sampler.sample_with_specific_counts_no_repeat(agent_type_counts)
            agent_configs, camera_configs, sample_center,sample_radius = self.sample_agent_configs(sampled_agent, vehicle_zones=vehicle_zones)
        else:
            num_agents = random.randint(self.min_total_agents, self.max_total_agents)
            sampled_agent, name_mapping_dict = self.agent_sampler.sample_agent_typid(agent_type_category=self.agent_category, agent_num=num_agents)
            env.refer_agents_category = list(sampled_agent.keys())
            env.num_agents = num_agents + len(env.player_list)
            agent_configs, camera_configs, sample_center,sample_radius = self.sample_agent_configs(sampled_agent, vehicle_zones=vehicle_zones)
        
        for agent_type,info_val in agent_configs.items(): # Renamed 'info' to 'info_val' to avoid conflict
            
            if agent_type == 'car' or agent_type == 'drone':
                for i in range(len(info_val['start_pos'])):
                    agent_configs[agent_type]['start_pos'][i][2] += height_bias[agent_type]
            elif agent_type == 'animal':
                for i in range(len(info_val['start_pos'])):
                    if info_val['app_id'][i] in [5,27]:
                        agent_configs[agent_type]['start_pos'][i][2] += height_bias[agent_type][1]
                    elif info_val['app_id'][i] == 19:  # Penguin test
                        agent_configs[agent_type]['start_pos'][i][2] += height_bias[agent_type][2]
                    else:
                        agent_configs[agent_type]['start_pos'][i][2] += height_bias[agent_type][0]
        
        print("agent_configs\n",agent_configs)
        print("camera_configs\n",camera_configs)
        self.agent_category = list(agent_configs.keys())
        # record env configs
        env.target_configs = agent_configs
        env.target_agents = misc.convert_dict(env.target_configs)
        print("target_agents\n",env.target_agents)
        env.camera_position = camera_configs['camera']['start_pos']
        env.name_mapping_dict = name_mapping_dict
        env.sampling_center = sample_center
        env.sampling_radius = sample_radius
        # set start point
        env.safe_start = [camera_configs['camera']['start_pos'][0]]
        
        # check agent type
        expect_agent_types = set(self.agent_category) if self.agent_category else set(env.agent_configs.keys())
        now_agent_types = set(env.agent_configs.keys())
        differ_agent_types = expect_agent_types - now_agent_types
        if differ_agent_types:  # if there are missing agent types
            # If there are missing agent types, we need to add them
            for agent_type in differ_agent_types:
                agent_config_2add = ConfigGenerator.add_agent_type(agent_type)
                env.agent_configs = env.agent_configs | agent_config_2add
                env.refer_agents = env.refer_agents | misc.convert_dict(agent_config_2add)
        env.set_population(env.num_agents)
        states,info_reset = self.env.reset(**kwargs) # Renamed 'info' to 'info_reset'
        return states,info_reset
    

    def sample_agent_types(self):
        """
        简化版：直接根据type_count_ranges为每种类型随机选择一个数值，筛除数目为0的类型
        """
        default_ranges_for_types = {
            'player': (1, 8),  
            'car': (0, 1),        
            'drone': (0, 0),      
            'motorbike': (0, 1),  
            'animal': (0, 0)      
        }
    
        # 使用默认值或传入的值
        current_type_count_ranges = self.type_count_ranges
        if current_type_count_ranges is None:
            print("No type_count_ranges provided, using default ranges.")
            current_type_count_ranges = default_ranges_for_types
        
        # 筛选可用类型（排除最大值为0的类型）
        available_types = []
        for agent_type in self.agent_category:
            min_count, max_count = current_type_count_ranges.get(agent_type, (0, 0))
            if max_count > 0:  # 只保留最大数量大于0的类型
                available_types.append(agent_type)
        
        if not available_types:
            print("警告: 所有类型的最大数量都为0，无法采样。")
            return {}
        
        # 为每种可用类型随机选择一个数量
        agents_type_counts = {}
        for agent_type in available_types:
            min_count, max_count = current_type_count_ranges.get(agent_type, (0, 0))
            # 随机选择一个范围内的值
            count = random.randint(min_count, max_count)
            if count > 0:  # 只添加数量大于0的类型
                agents_type_counts[agent_type] = count
        
        # 确保总数在允许范围内
        total_agents = sum(agents_type_counts.values())
        if total_agents < self.min_total_agents:
            # 如果总数少于最小要求，增加一些智能体
            deficit = self.min_total_agents - total_agents
            eligible_types = []
            for agent_type in agents_type_counts:
                min_count, max_count = current_type_count_ranges.get(agent_type, (0, 0))
                current_count = agents_type_counts[agent_type]
                if current_count < max_count:
                    eligible_types.append({
                        'type': agent_type,
                        'current': current_count,
                        'max': max_count,
                        'space': max_count - current_count
                    })
            
            # 按照权重分配剩余智能体
            while deficit > 0 and eligible_types:
                # 按剩余空间排序
                eligible_types.sort(key=lambda x: x['space'], reverse=True)
                chosen_type = eligible_types[0]['type']
                agents_type_counts[chosen_type] += 1
                eligible_types[0]['current'] += 1
                eligible_types[0]['space'] -= 1
                deficit -= 1
                
                # 如果没有剩余空间，移除该类型
                if eligible_types[0]['current'] >= eligible_types[0]['max']:
                    eligible_types.pop(0)
        
        elif total_agents > self.max_total_agents:
            # 如果总数多于最大允许，减少一些智能体
            excess = total_agents - self.max_total_agents
            types_to_reduce = list(agents_type_counts.keys())
            random.shuffle(types_to_reduce)  # 随机排序，以便随机减少
            
            for agent_type in types_to_reduce:
                if excess <= 0:
                    break
                
                current_count = agents_type_counts[agent_type]
                min_count, _ = current_type_count_ranges.get(agent_type, (0, 0))
                can_reduce = max(0, current_count - min_count)
                reduction = min(can_reduce, excess)
                
                agents_type_counts[agent_type] -= reduction
                excess -= reduction
                
                # 如果减少后数量为0，移除该类型
                if agents_type_counts[agent_type] == 0:
                    agents_type_counts.pop(agent_type)
        
        # 移除数量为0的类型
        agents_type_counts = {k: v for k, v in agents_type_counts.items() if v > 0}
        
        # 打印结果
        total_agents = sum(agents_type_counts.values())
        print(f"采样配置 - 选定类型: {list(agents_type_counts.keys())}")
        print(f"采样配置 - 总智能体数: {total_agents} (范围: {self.min_total_agents}-{self.max_total_agents})")
        print(f"采样配置 - 详细分布: {agents_type_counts}")
        
        return agents_type_counts

    def sample_agent_configs(self,agent_configs, vehicle_zones=None):
        all_max_distance = self.point_sampler.compute_adaptive_all_max_distance(agent_configs)
        max_retry = 200
        for attempt in range(max_retry):
            results_dict = self.point_sampler.sample_for_predefined_agents( 
                agent_configs=agent_configs,
                camera_count=8,
                vehicle_zones=vehicle_zones,
                all_max_distance=all_max_distance,
                ring_inner_radius_offset=300, 
                ring_outer_radius_offset=500,
                min_angle_separation_deg=35,
                min_cam_to_agent_dist=200 
            )
            updated_configs = results_dict['agent_configs']
            camera_configs = results_dict['camera_configs']
            agent_sampling_center = results_dict['sampling_center']
            agent_sampling_radius = results_dict['sampling_radius']
            # sampling_results = results_dict
            # 检查是否所有物体都采样成功
            total_needed = sum(len(cfg['name']) for cfg in agent_configs.values())
            total_found = sum(len(cfg['name']) for cfg in updated_configs.values())
            if total_found == total_needed:
                print(f"采样成功（第{attempt+1}次）")
                break
            else:
                print(f"采样失败，重新尝试（第{attempt+1}次）")
        else:
            print("多次尝试后仍未采样成功，请检查参数或环境！")
        return updated_configs, camera_configs,agent_sampling_center, agent_sampling_radius