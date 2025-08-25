import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

def merge_pkl_files(file1, file2, output_file):
    """
    合并两个.pkl文件中的可达点数据
    
    Args:
        file1: 第一个.pkl文件路径
        file2: 第二个.pkl文件路径
        output_file: 输出文件路径
    """
    # 加载第一个文件
    print(f"加载文件: {file1}")
    with open(file1, 'rb') as f:
        data1 = pickle.load(f)
    
    # 加载第二个文件
    print(f"加载文件: {file2}")
    with open(file2, 'rb') as f:
        data2 = pickle.load(f)
    
    # 检查数据结构
    if not isinstance(data1, dict) or not isinstance(data2, dict):
        print("警告: 文件格式不是预期的字典结构，合并可能不准确")
    
    # 创建合并后的数据字典
    merged_data = {}
    
    # 合并可达点
    if 'reachable_points' in data1 and 'reachable_points' in data2:
        merged_points = {**data1['reachable_points'], **data2['reachable_points']}
        merged_data['reachable_points'] = merged_points
        print(f"合并后的可达点数量: {len(merged_points)}")
    else:
        print("警告: 找不到可达点数据")
        
        # 尝试直接合并顶层字典(如果没有reachable_points字段)
        merged_data = {**data1, **data2}
    
    # 合并智能体路径
    if 'agent_paths' in data1 and 'agent_paths' in data2:
        # 对于路径，我们保留两个文件中的所有路径
        merged_paths = data1['agent_paths'] + data2['agent_paths']
        merged_data['agent_paths'] = merged_paths
        print(f"合并后的路径数量: {len(merged_paths)}")
    
    # 保留其他元数据
    for key in set(data1.keys()) | set(data2.keys()):
        if key not in ['reachable_points', 'agent_paths']:
            # 对于冲突的值，优先使用第一个文件的
            merged_data[key] = data1.get(key, data2.get(key))
    
    # 保存合并后的数据
    print(f"保存合并后的数据到: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(merged_data, f)
    
    return merged_data

def visualize_merged_data(merged_data, output_image="merged_points.png"):
    """可视化合并后的数据"""
    if 'reachable_points' not in merged_data:
        print("警告: 合并数据中没有可达点字段，无法可视化")
        return
    
    # 提取所有位置点
    points = np.array(list(merged_data['reachable_points'].values()))
    
    plt.figure(figsize=(12, 10))
    
    # 绘制可达点
    plt.scatter(points[:, 0], points[:, 1], c='blue', s=1, alpha=0.5)
    
    # 绘制路径(如果有)
    if 'agent_paths' in merged_data:
        colors = ['r', 'g', 'm', 'y', 'c', 'orange', 'purple', 'brown']
        for i, path in enumerate(merged_data['agent_paths']):
            if path:
                path_array = np.array(path)
                plt.plot(path_array[:, 0], path_array[:, 1], 
                        color=colors[i % len(colors)], 
                        linewidth=0.5, alpha=0.3)
    
    plt.title(f"合并后的可达点 ({len(merged_data['reachable_points'])}个点)")
    plt.xlabel("X 坐标")
    plt.ylabel("Y 坐标")
    
    # 保存图像
    plt.savefig(output_image, dpi=300)
    print(f"可视化结果已保存为: {output_image}")
    plt.close()

if __name__ == "__main__":
    # 使用示例
    file1 = "E:\\EQA\\unrealzoo_gym\\example\\agent_configs_sampler\\points_graph\\Map_ChemicalPlant_1\\Map_ChemicalPlant_1_final.pkl"
    file2 = "E:\\EQA\\unrealzoo_gym\\example\\agent_configs_sampler\\points_graph\\Map_ChemicalPlant_1\\Map_ChemicalPlant_1.pkl"
    output_file = "E:\\EQA\\unrealzoo_gym\\example\\agent_configs_sampler\\points_graph\\Map_ChemicalPlant_1\\Map_ChemicalPlant_1_merged.pkl"

    # 合并文件
    merged_data = merge_pkl_files(file1, file2, output_file)
    
    # 可视化合并结果
    visualize_merged_data(merged_data, "merged_coverage_map.png")