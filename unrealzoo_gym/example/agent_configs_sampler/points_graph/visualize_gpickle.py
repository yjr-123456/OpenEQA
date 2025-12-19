import pickle
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 注册3D投影
import os

def visualize_graph_3d(graph, output_file="connectivity_graph_3d.svg", show_largest_component=True):
    """
    在三维空间中可视化给定的 networkx 图。

    Args:
        graph (nx.Graph): 要可视化的图对象。
        output_file (str): 输出图像文件的路径。
        show_largest_component (bool): 是否高亮显示最大的连通分量。
    """
    if not isinstance(graph, nx.Graph):
        print("错误：提供的对象不是一个有效的 networkx 图。")
        return

    # --- 1. 设置3D绘图环境 ---
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    # --- 2. 准备节点坐标 ---
    # 获取所有节点的真实三维坐标
    pos_3d = nx.get_node_attributes(graph, 'pos')
    if not pos_3d:
        print("错误：图中节点缺少 'pos' 属性，无法进行3D可视化。")
        return
        
    # 将坐标解包为 x, y, z 列表
    nodes = list(graph.nodes())
    x_coords = [pos_3d[n][0] for n in nodes]
    y_coords = [pos_3d[n][1] for n in nodes]
    z_coords = [pos_3d[n][2] for n in nodes]

    # --- 3. 绘制所有节点 ---
    ax.scatter(x_coords, y_coords, z_coords, c='skyblue', s=10, alpha=0.5, label='Other Nodes')

    # --- 4. 绘制所有边 ---
    for edge in graph.edges():
        p1 = pos_3d[edge[0]]
        p2 = pos_3d[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='gray', alpha=0.2, linewidth=0.5)

    # --- 5. 高亮显示最大的连通分量 ---
    if show_largest_component and len(graph) > 0:
        try:
            largest_cc = max(nx.connected_components(graph), key=len)
            
            # 获取最大连通分量节点的坐标
            cc_x = [pos_3d[n][0] for n in largest_cc]
            cc_y = [pos_3d[n][1] for n in largest_cc]
            cc_z = [pos_3d[n][2] for n in largest_cc]
            
            # 用不同颜色重新绘制这些节点
            ax.scatter(cc_x, cc_y, cc_z, c='red', s=15, alpha=0.7, label='Largest Component')
        except (ValueError, nx.NetworkXError):
            print("警告：图中没有节点或无法找到连通分量。")


    # --- 6. 设置图形属性并保存 ---
    ax.set_title(f"3D Environment Connectivity Graph ({len(graph.nodes())} nodes, {len(graph.edges())} edges)")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.legend()
    ax.axis('off')
    # 调整视角
    ax.view_init(elev=30, azim=45)

    plt.savefig(output_file, format='svg', bbox_inches='tight')
    print(f"3D连接图已保存到 {output_file}")
    plt.close()

# --- 使用示例 ---
if __name__ == "__main__":
    # --- 配置 ---
    # 1. 指定环境名称
    env_name = "SuburbNeighborhood_Day"
    
    # 2. 构建.gpickle文件的完整路径
    gpickle_file = os.path.join(f"./{env_name}", "environment_graph_1.gpickle")
    
    # 3. 指定输出的SVG文件名
    output_svg_file = os.path.join(f"./{env_name}", f"{env_name}_graph_3d_from_gpickle.svg")

    # --- 执行 ---
    if not os.path.exists(gpickle_file):
        print(f"错误：找不到文件 '{gpickle_file}'。请确保文件路径正确或先运行make_graph.py生成该文件。")
    else:
        print(f"正在加载图文件: {gpickle_file}")
        # 加载图文件
        with open(gpickle_file, 'rb') as f:
            loaded_graph = pickle.load(f)
        
        print(f"加载成功！图中包含 {loaded_graph.number_of_nodes()} 个节点和 {loaded_graph.number_of_edges()} 条边。")
        
        # 调用可视化函数
        visualize_graph_3d(loaded_graph, output_svg_file)