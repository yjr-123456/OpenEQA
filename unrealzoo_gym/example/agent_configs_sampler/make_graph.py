import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
#from networkx.readwrite.gpickle import write_gpickle

class TrajectoryGraphBuilder:
    def __init__(self, pickle_file_path):
        """
        Load trajectory data from saved pkl file and build a connectivity graph
        
        Args:
            pickle_file_path: Path to the saved trajectory points file
        """
        # Load saved data
        with open(pickle_file_path, 'rb') as f:
            data = pickle.load(f)
        
        self.reachable_grid = data["reachable_points"]  # Dictionary of grid-based reachable points
        self.agent_paths = data["agent_paths"]  # Agent trajectories
        self.resolution = data["resolution"]  # Grid resolution
        
        print(f"Loaded {len(self.reachable_grid)} reachable points and {len(self.agent_paths)} agent trajectories")
        
        # Initialize graph structure
        self.graph = None
        
    def build_graph(self, distance_threshold=150.0, min_node_distance=100.0):
        """
        Build connectivity graph with node filtering
        
        Args:
            distance_threshold: Maximum distance threshold for two points to be considered connected
            min_node_distance: Minimum distance between nodes (points closer than this will be filtered)
        """
        G = nx.Graph()
        
        # Filter reachable points to keep only those that are sufficiently far apart
        filtered_points = {}
        points_list = list(self.reachable_grid.items())
        
        print(f"Starting with {len(points_list)} potential nodes...")
        
        # Add first point as reference
        if points_list:
            grid_pos, real_pos = points_list[0]
            filtered_points[grid_pos] = real_pos
            
        # Check remaining points against already filtered ones
        for grid_pos, real_pos in points_list[1:]:
            too_close = False
            
            # Check if this point is too close to any already accepted point
            for existing_pos in filtered_points.values():
                dist = np.linalg.norm(np.array(real_pos) - np.array(existing_pos))
                if dist < min_node_distance:
                    too_close = True
                    break
                    
            # If not too close to any existing point, add it
            if not too_close:
                filtered_points[grid_pos] = real_pos
        
        print(f"After filtering, keeping {len(filtered_points)} nodes " +
              f"(removed {len(points_list) - len(filtered_points)} nodes)")
        
        # Add filtered points as nodes
        for grid_pos, real_pos in filtered_points.items():
            G.add_node(grid_pos, pos=real_pos)
        
        # Add edges from agent trajectories - connect adjacent time steps
        edges_added = set()
        for path in self.agent_paths:
            for i in range(len(path) - 1):
                # Get grid coordinates for adjacent positions
                pos1 = self._discretize_position(path[i])
                pos2 = self._discretize_position(path[i+1])
                
                # Skip if either position was filtered out
                if pos1 not in filtered_points or pos2 not in filtered_points:
                    continue
                    
                # Avoid duplicate edges
                edge_key = tuple(sorted([pos1, pos2]))
                if edge_key in edges_added:
                    continue
                
                # Only add edge when two points are different and distance is within threshold
                if pos1 != pos2:
                    dist = np.linalg.norm(np.array(path[i]) - np.array(path[i+1]))
                    if dist <= distance_threshold:
                        G.add_edge(pos1, pos2, weight=dist)
                        edges_added.add(edge_key)
        
        # Add additional edges based on spatial proximity
        points = list(filtered_points.keys())
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                p1, p2 = points[i], points[j]
                # Avoid duplicate edges
                edge_key = tuple(sorted([p1, p2]))
                if edge_key in edges_added:
                    continue
                
                # Calculate real position distance
                dist = np.linalg.norm(np.array(filtered_points[p1]) - np.array(filtered_points[p2]))
                
                # Add edge if distance is within threshold
                if dist <= distance_threshold:
                    G.add_edge(p1, p2, weight=dist)
                    edges_added.add(edge_key)
        
        self.graph = G
        print(f"Built connectivity graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
        
        # Rest of your existing code...
        # Analyze connected components
        components = list(nx.connected_components(G))
        print(f"Graph has {len(components)} connected components")
        for i, component in enumerate(sorted(components, key=len, reverse=True)):
            print(f"  Component {i+1}: {len(component)} nodes ({100*len(component)/len(G.nodes()):.1f}% coverage)")
            if i >= 4:  # Only show top 5 largest components
                break
        
        return G       

    def visualize_graph(self, output_file="connectivity_graph.png", show_largest_component=True):
        """
        Visualize connectivity graph
        
        Args:
            output_file: Output image file path
            show_largest_component: Whether to highlight the largest connected component
        """
        if self.graph is None:
            print("Please call build_graph() first to build the graph")
            return
            
        plt.figure(figsize=(12, 12))
        
        # Create node position dictionary (projected to 2D plane)
        pos = {node: (self.reachable_grid[node][0], self.reachable_grid[node][1]) for node in self.graph.nodes()}
        
        # Use degree centrality for node size
        node_sizes = [1 + 3 * self.graph.degree(node) for node in self.graph.nodes()]
        
        # Draw all nodes and edges
        nx.draw_networkx_nodes(self.graph, pos, node_size=node_sizes, 
                              node_color='skyblue', alpha=0.6)
        nx.draw_networkx_edges(self.graph, pos, width=0.3, 
                              edge_color='gray', alpha=0.3)
        
        # Highlight the largest connected component
        if show_largest_component:
            largest_cc = max(nx.connected_components(self.graph), key=len)
            subgraph = self.graph.subgraph(largest_cc)
            subgraph_node_sizes = [1 + 3 * self.graph.degree(node) for node in subgraph.nodes()]
        
            nx.draw_networkx_nodes(subgraph, pos, node_size=subgraph_node_sizes,
                              node_color='red', alpha=0.4)
            
        plt.title(f"Environment Connectivity Graph: {len(self.graph.nodes())} nodes, {len(self.graph.edges())} edges")
        plt.axis('off')
        plt.tight_layout()
        
        # Save image
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Connectivity graph saved to {output_file}")
        plt.close()
    
    def visualize_graph_3d(self, output_file="connectivity_graph_3d.png", show_largest_component=True):
            """
            在三维空间中可视化连接图。

            Args:
                output_file: 输出图像文件的路径。
                show_largest_component: 是否高亮显示最大的连通分量。
            """
            if self.graph is None:
                print("请先调用 build_graph() 来构建图。")
                return

            # --- 2. 设置3D绘图环境 ---
            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(111, projection='3d')

            # --- 3. 准备节点坐标 ---
            # 获取所有节点的真实三维坐标
            pos_3d = nx.get_node_attributes(self.graph, 'pos')
            if not pos_3d:
                print("错误：图中节点没有 'pos' 属性。")
                return
                
            # 将坐标解包为 x, y, z 列表
            nodes = list(self.graph.nodes())
            x_coords = [pos_3d[n][0] for n in nodes]
            y_coords = [pos_3d[n][1] for n in nodes]
            z_coords = [pos_3d[n][2] for n in nodes]

            # --- 4. 绘制所有节点 ---
            ax.scatter(x_coords, y_coords, z_coords, c='skyblue', s=10, alpha=0.5, label='Other Nodes')

            # --- 5. 绘制所有边 ---
            for edge in self.graph.edges():
                p1 = pos_3d[edge[0]]
                p2 = pos_3d[edge[1]]
                # 使用 plot 绘制连接两个点的线段
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='gray', alpha=0.2, linewidth=0.5)

            # --- 6. 高亮显示最大的连通分量 ---
            if show_largest_component and len(self.graph) > 0:
                largest_cc = max(nx.connected_components(self.graph), key=len)
                
                # 获取最大连通分量节点的坐标
                cc_x = [pos_3d[n][0] for n in largest_cc]
                cc_y = [pos_3d[n][1] for n in largest_cc]
                cc_z = [pos_3d[n][2] for n in largest_cc]
                
                # 用不同颜色重新绘制这些节点
                ax.scatter(cc_x, cc_y, cc_z, c='red', s=15, alpha=0.7, label='Largest Component')

            # --- 7. 设置图形属性并保存 ---
            ax.set_title(f"3D Environment Connectivity Graph ({len(self.graph.nodes())} nodes, {len(self.graph.edges())} edges)")
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
            exit(-1)
    def find_path(self, start_pos, end_pos):
        """
        Find a path from start to end position in the graph
        
        Args:
            start_pos: Starting position coordinates (x,y,z)
            end_pos: Target position coordinates (x,y,z)
            
        Returns:
            path: List of path points
        """
        if self.graph is None:
            print("Please call build_graph() first to build the graph")
            return []
            
        # Convert to grid coordinates
        start_grid = self._discretize_position(start_pos)
        end_grid = self._discretize_position(end_pos)
        
        # Ensure points are in the graph
        if start_grid not in self.graph:
            # Find the nearest starting point
            start_grid = min(self.graph.nodes(), key=lambda n: 
                           np.linalg.norm(np.array(self.reachable_grid[n]) - np.array(start_pos)))
            print(f"Start point not in graph, using nearest point: {self.reachable_grid[start_grid]}")
            
        if end_grid not in self.graph:
            # Find the nearest end point
            end_grid = min(self.graph.nodes(), key=lambda n: 
                         np.linalg.norm(np.array(self.reachable_grid[n]) - np.array(end_pos)))
            print(f"End point not in graph, using nearest point: {self.reachable_grid[end_grid]}")
            
        # Check if two points are in the same connected component
        if not nx.has_path(self.graph, start_grid, end_grid):
            print("Start and end points are not in the same connected component, cannot find path")
            return []
            
        # Use Dijkstra's algorithm to find shortest path
        path = nx.shortest_path(self.graph, start_grid, end_grid, weight='weight')
        
        # Convert to real coordinates
        real_path = [self.reachable_grid[p] for p in path]
        
        # Calculate total path length
        total_length = sum(self.graph[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
        
        print(f"Found path: {len(path)} steps, total length: {total_length:.2f}")
        return real_path
        
    def visualize_path(self, path, output_file="path.png"):
        """
        Visualize path
        
        Args:
            path: List of path points
            output_file: Output image file path
        """
        if not path:
            print("Path is empty")
            return
            
        plt.figure(figsize=(12, 12))
        
        # Draw all nodes
        points = np.array(list(self.reachable_grid.values()))
        plt.scatter(points[:, 0], points[:, 1], c='lightgray', s=1, alpha=0.3)
        
        # Draw path
        path_array = np.array(path)
        plt.plot(path_array[:, 0], path_array[:, 1], 'r-', linewidth=2)
        
        # Mark start and end points
        plt.scatter(path_array[0, 0], path_array[0, 1], c='g', s=100, marker='o', label='Start')
        plt.scatter(path_array[-1, 0], path_array[-1, 1], c='r', s=100, marker='x', label='End')
        
        plt.title(f"Path: {len(path)} steps")
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(output_file, dpi=300)
        print(f"Path saved to {output_file}")
        plt.close()
    
    def analyze_graph(self,env_name):
        """
        Analyze topological characteristics of the graph
        """
        if self.graph is None:
            print("Please call build_graph() first to build the graph")
            return
            
        # Calculate basic graph properties
        print("\n===== Graph Analysis =====")
        print(f"Number of nodes: {len(self.graph.nodes())}")
        print(f"Number of edges: {len(self.graph.edges())}")
        
        # Connectivity analysis
        components = list(nx.connected_components(self.graph))
        print(f"Number of connected components: {len(components)}")
        
        largest_cc = max(components, key=len)
        print(f"Largest connected component size: {len(largest_cc)} nodes ({100*len(largest_cc)/len(self.graph.nodes()):.1f}%)")
        
        # Graph diameter (maximum shortest path length)
        largest_cc_graph = self.graph.subgraph(largest_cc)
        diameter = nx.diameter(largest_cc_graph)
        print(f"Diameter of largest connected component: {diameter}")
        
        # Calculate node importance
        print("\nTop 5 important nodes (based on degree centrality):")
        degree_centrality = nx.degree_centrality(self.graph)
        top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (node, centrality) in enumerate(top_nodes):
            print(f"  {i+1}. Node {node}: centrality = {centrality:.4f}, real coordinates = {self.reachable_grid[node]}")
            
        # Save graph data
        with open(f"./points_graph/{env_name}/environment_graph_1.gpickle", 'wb') as f:
            pickle.dump(self.graph, f)
        # nx.gpickle.write_gpickle(self.graph, "environment_graph.gpickle")
        print("\nGraph saved to environment_graph_1.gpickle")

    def _discretize_position(self, position):
        """Discretize continuous position into grid coordinates"""
        x, y, z = position
        grid_x = round(x / self.resolution)
        grid_y = round(y / self.resolution)
        grid_z = round(z / self.resolution)
        return (grid_x, grid_y, grid_z)


# Usage example
if __name__ == "__main__":
    # Replace this path with your pkl file path
    env_list = [
        "FlexibleRoom",
    # "Map_ChemicalPlant_1",
    # "ModularNeighborhood",
    # "ModularSciFiVillage",
    # "RuralAustralia_Example_01",
    # "ModularVictorianCity",
    # "Cabin_Lake",
    # "Pyramid",
    # "ModularGothic_Day",
    # "Greek_Island",
    # "PlanetOutDoor",
    # "AsianMedivalCity"
    ]
    for env_name in env_list:
        pickle_file = f"E:/EQA/unrealzoo_gym/example/agent_configs_sampler/points_graph/{env_name}/{env_name}_reachable_points.pkl"

        # Create graph builder
        builder = TrajectoryGraphBuilder(pickle_file)

        # Build connectivity graph
        graph = builder.build_graph(distance_threshold=200.0,min_node_distance= 50.0)  # Adjust distance threshold to suit your environment scale
        
        # Visualize graph
        # builder.visualize_graph(f"{env_name}_graph.png")
        # builder.visualize_graph_3d(f"{env_name}_graph_3d.svg")
        # builder.visualize_graph(f"{env_name}_graph.png")
        # Analyze graph
        builder.analyze_graph(env_name=env_name)

    # Path planning example
    # Note: Replace with valid coordinates in your environment
    # start_pos = [0, 0, 100]  # Example starting point
    # end_pos = [1000, 1000, 100]  # Example end point
    # path = builder.find_path(start_pos, end_pos)
    # if path:
    #     builder.visualize_path(path, "example_path.png")