import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
# plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# --- 1. 配置和初始化 ---

# 数据根目录
base_data_dir = "E:/EQA/unrealzoo_gym/example/QA_data_sub"

# 环境列表
env_list = [
    "ModularNeighborhood", "Map_ChemicalPlant_1", "Pyramid", "Greek_Island",
    "SuburbNeighborhood_Day", "LV_Bazaar", "DowntownWest", "PlanetOutDoor",
    "RussianWinterTownDemo01", "AsianMedivalCity", "Medieval_Castle",
    "SnowMap", "Real_Landscape", "Demonstration_Castle", "Venice"
]

# 问题类型列表
question_types = ["counting", "relative_location", "relative_distance", "state"]

# 用于存储统计结果的字典
# defaultdict 会在键不存在时提供一个默认值（这里是0）
question_counts = defaultdict(int)
agent_counts = defaultdict(int)
scene_counts_per_map = defaultdict(int)
total_questions = 0

# --- 2. 高效地一次性遍历并统计所有数据 ---

print("开始统计数据...")

for env_name in env_list:
    env_path = os.path.join(base_data_dir, env_name)
    if not os.path.isdir(env_path):
        print(f"警告: 环境目录不存在，跳过 -> {env_path}")
        continue

    # 获取当前环境下的所有场景文件夹
    scene_folders = [d for d in os.listdir(env_path) if os.path.isdir(os.path.join(env_path, d))]
    scene_counts_per_map[env_name] = len(scene_folders)

    for scene_folder in scene_folders:
        scene_path = os.path.join(env_path, scene_folder)
        qa_file_path = os.path.join(scene_path, "qa_data.json")

        if not os.path.isfile(qa_file_path):
            continue

        try:
            with open(qa_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 统计问题数量
            if "Questions" in data:
                for q_type, questions in data["Questions"].items():
                    count = len(questions)
                    question_counts[q_type] += count
                    total_questions += count
            
            # 统计 Agent 数量
            if "target_configs" in data:
                for agent_type, config in data["target_configs"].items():
                    # 确保 'name' 键存在且是一个列表
                    agent_counts[agent_type] += len(config.get("name", []))

        except (json.JSONDecodeError, KeyError) as e:
            print(f"错误: 处理文件 {qa_file_path} 时出错: {e}")

print("数据统计完成！\n")

# --- 3. 打印统计结果 ---

print("--- 问题类型统计 ---")
for q_type, count in question_counts.items():
    print(f"{q_type}: {count} 个")
print(f"总问题数: {total_questions}\n")

print("--- Agent 类型统计 ---")
for agent_type, count in agent_counts.items():
    print(f"{agent_type.capitalize()}: {count} 个")
print("")

print("--- 各地图场景数统计 ---")
for env_name, count in scene_counts_per_map.items():
    print(f"{env_name}: {count} 个场景")
print("")


# --- 4. 绘制三个饼状图 ---

def plot_pie_chart(data_dict, title):
    """一个通用的函数，用于绘制饼状图并保存为矢量图"""
    labels = list(data_dict.keys())
    sizes = list(data_dict.values())

    # 过滤掉数量为0的项
    non_zero_labels = [label for i, label in enumerate(labels) if sizes[i] > 0]
    non_zero_sizes = [size for size in sizes if size > 0]

    if not non_zero_sizes:
        print(f"没有为 '{title}' 找到可供绘图的数据。")
        return

    # --- 修改：增加图形尺寸 ---
    fig, ax = plt.subplots(figsize=(12, 8)) # 将宽度从10增加到12
    wedges, texts, autotexts = ax.pie(
        non_zero_sizes, 
        labels=non_zero_labels, 
        autopct='%1.1f%%', 
        startangle=90,
        textprops=dict(color="w") # 让百分比文字为白色
    )
    
    # 在图例中显示具体数值
    legend_labels = [f'{l} ({s})' for l, s in zip(non_zero_labels, non_zero_sizes)]
    ax.legend(wedges, legend_labels, title="类别", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title(title)
    
    # --- 新增：保存为 SVG 矢量图 ---
    # 1. 根据标题创建一个安全的文件名 (例如: "distribution_of_question_types.svg")
    safe_filename = title.lower().replace(' ', '_') + ".svg"
    
    # 2. 保存图形，bbox_inches='tight'确保图例等内容不会被裁剪
    plt.savefig(safe_filename, format='svg', bbox_inches='tight')
    print(f"矢量图已保存为: {safe_filename}")

    # 3. 仍然在屏幕上显示图形
    plt.show()


# 绘制第一个图：问题类型分布
plot_pie_chart(question_counts, "Distribution of Question Types")

# 绘制第二个图：Agent 类型分布
plot_pie_chart(agent_counts, "Distribution of Agent Types")

# 绘制第三个图：每个地图的场景数分布
plot_pie_chart(scene_counts_per_map, "Distribution of Scenes per Map")