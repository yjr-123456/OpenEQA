from gymnasium import Wrapper
from example.agent_configs_sampler import AgentSampler, AgentBasedSampler
import random
from gym_unrealcv.envs.utils import misc
import time
from gym_unrealcv.envs.wrappers.augmentation import ConfigGenerator
class SampleAgentConfigWrapper(Wrapper):
    def __init__(self, env, agent_category, camera_height, model, 
                 min_types, max_types, type_count_ranges,
                 min_total_agents, max_total_agents, 
                 graph_path, config_path, obj_2_hide=None,
                 use_adaptive=False, if_cnt=False,
                 fast_test_mode=False,
                 save_dir=None,
                 normal_variance_threshold=0.05,
                 slope_threshold=0.866,
                 safety_margin_cm=50,
                 gaussian_kernel_size=5,
                 gaussian_sigma=1.0):
        """
        初始化采样配置Wrapper
        
        Args:
            ... (原有参数)
            use_adaptive: 是否使用自适应采样器
            fast_test_mode: 是否使用快速测试模式（随机选点）
            save_dir: 调试图像保存目录
            normal_variance_threshold: 法线方差阈值
            slope_threshold: 坡度阈值
            safety_margin_cm: 安全边距（厘米）
            gaussian_kernel_size: 高斯核大小
            gaussian_sigma: 高斯sigma
        """
        super().__init__(env)
        
        # 原有参数
        self.agent_category = agent_category
        self.camera_height = camera_height
        self.model = model
        self.min_types = min_types
        self.max_types = max_types
        self.type_count_ranges = type_count_ranges
        self.min_total_agents = min_total_agents
        self.max_total_agents = max_total_agents
        self.graph_path = graph_path
        self.config_path = config_path
        self.obj_2_hide = obj_2_hide or []
        self.use_adaptive = use_adaptive
        self.if_cnt = if_cnt

        self.fast_test_mode = fast_test_mode
        self.save_dir = save_dir or './test_results/'
        self.normal_variance_threshold = normal_variance_threshold
        self.slope_threshold = slope_threshold
        self.safety_margin_cm = safety_margin_cm
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_sigma = gaussian_sigma
        
        # 创建采样器
        self._init_sampler()
        
        # 初始化采样器
        self.agent_sampler = AgentSampler()
        if self.use_adaptive:
            from example.agent_configs_sampler.agent_sample_agent_advanced import AgentBasedSamplerboost
            self.agent_point_sampler = AgentBasedSamplerboost(graph_path, model, config_path=config_path)
        else:
            self.agent_point_sampler = AgentBasedSampler(graph_path, model, config_path=config_path)
    
    def step(self,action):
        obs, reward, termination,truncation, info = self.env.step(action)
        return obs, reward, termination,truncation, info

    def _init_sampler(self):
        """初始化采样器"""
        if self.use_adaptive:
            from example.agent_configs_sampler.agent_sample_agent_advanced import AgentBasedSamplerboost
            self.agent_point_sampler = AgentBasedSamplerboost(
                graph_pickle_file=self.graph_path,
                model=self.model,
                config_path=self.config_path
            )
        else:
            from example.agent_configs_sampler import AgentBasedSampler
            self.agent_point_sampler = AgentBasedSampler(
                graph_pickle_file=self.graph_path,
                model=self.model,
                config_path=self.config_path
            )
    
    def update_sampling_params(self, **kwargs):
        """
        动态更新采样参数
        
        支持的参数：
            - normal_variance_threshold
            - slope_threshold
            - safety_margin_cm
            - gaussian_kernel_size
            - gaussian_sigma
            - fast_test_mode
            - save_dir
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"[Wrapper] 更新参数: {key} = {value}")
    

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
        # hide some objs
        if self.obj_2_hide is not None:
            env.unrealcv.set_hide_objects(self.obj_2_hide)
        # sample agent num for each agent type
        if self.if_cnt == False:
            agent_type_counts = self.sample_agent_types() 
            print("agent_type_counts\n",agent_type_counts)
            env.refer_agents_category = list(agent_type_counts.keys())        
            env.num_agents = sum(agent_type_counts.values()) + len(env.player_list)
            sampled_agent, name_mapping_dict = self.agent_sampler.sample_with_specific_counts_no_repeat(agent_type_counts)
            env, agent_configs, camera_configs, sample_center,sample_radius = self.sample_agent_configs(env,sampled_agent, cam_id=1,vehicle_zones=vehicle_zones)
        else:
            num_agents = random.randint(self.min_total_agents, self.max_total_agents)
            sampled_agent, name_mapping_dict = self.agent_sampler.sample_agent_typid(agent_type_category=self.agent_category, agent_num=num_agents)
            env.refer_agents_category = list(sampled_agent.keys())
            env.num_agents = num_agents + len(env.player_list)
            env, agent_configs, camera_configs, sample_center,sample_radius = self.sample_agent_configs(env, sampled_agent, cam_id=1, vehicle_zones=vehicle_zones)

        # show object
        if self.obj_2_hide is not None:
            env.unrealcv.set_show_objects(self.obj_2_hide)
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

    def sample_agent_configs(self, env, agent_configs, cam_id=1, cam_count=8, vehicle_zones=None):
        """
        采样智能体配置
        """
        all_max_distance = self.agent_point_sampler.compute_adaptive_all_max_distance(agent_configs)
        max_retry = 200
        sampling_kwargs = {
            'normal_variance_threshold': self.normal_variance_threshold,
            'slope_threshold': self.slope_threshold,
            'safety_margin_cm': self.safety_margin_cm,
            'gaussian_kernel_size': self.gaussian_kernel_size,
            'gaussian_sigma': self.gaussian_sigma,
            'fast_test_mode': self.fast_test_mode,
            'save_dir': self.save_dir,
            'ring_inner_radius_offset': 300,
            'ring_outer_radius_offset': 500,
            'min_angle_separation_deg': 35,
            'min_cam_to_agent_dist': 200,
        }



        for attempt in range(max_retry):
            # ===== 关键修改：使用保存的参数 =====
            results_dict = self.agent_point_sampler.sample_agent_positions(
                env=self.env.unwrapped,
                agent_configs=agent_configs,
                cam_id=1,
                cam_count=3,
                vehicle_zones=vehicle_zones,
                all_max_distance=all_max_distance,
                height=self.camera_height,
                **sampling_kwargs  # ← 传递所有参数
            )
            
            env = results_dict['env']
            updated_configs = results_dict['agent_configs']
            camera_configs = results_dict['camera_configs']
            agent_sampling_center = results_dict['sampling_center']
            agent_sampling_radius = results_dict['sampling_radius']
            
            # 检查是否所有物体都采样成功
            total_needed = sum(len(cfg['name']) for cfg in agent_configs.values())
            total_found = sum(len(cfg['name']) for cfg in updated_configs.values())
            
            if total_found == total_needed:
                print(f"✅ 采样成功（第{attempt+1}次）")
                break
            else:
                print(f"⚠️  采样失败，重新尝试（第{attempt+1}次）")
        else:
            print("❌ 多次尝试后仍未采样成功，请检查参数或环境！")
        
        return env, updated_configs, camera_configs, agent_sampling_center, agent_sampling_radius