
import numpy as np
import networkx as nx
from scipy.spatial import distance
from scipy.spatial.transform import Rotation
from typing import List, Dict, Tuple

# 定义语义节点类
class SemanticNode:
    def __init__(self, id: int, label: str, position: np.ndarray, descriptor: np.ndarray):
        self.id = id
        self.label = label
        self.position = position
        self.descriptor = descriptor

# 构建语义节点连接图
def build_semantic_graph(nodes: List[SemanticNode], distance_threshold: float = 2.0) -> nx.Graph:
    G = nx.Graph()
    for node in nodes:
        G.add_node(node.id, label=node.label, position=node.position, descriptor=node.descriptor)
    
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i < j:
                dist = distance.euclidean(node_i.position, node_j.position)
                if dist < distance_threshold:
                    G.add_edge(node_i.id, node_j.id, weight=dist)
    return G

# 从当前观测生成子图
def generate_observation_subgraph(observed_objects: List[Dict]) -> nx.Graph:
    nodes = []
    for i, obj in enumerate(observed_objects):
        node = SemanticNode(
            id=i,
            label=obj['label'],
            position=np.array(obj['position']),
            descriptor=np.array(obj['descriptor'])
        )
        nodes.append(node)
    return build_semantic_graph(nodes)

# 计算刚体变换（Umeyama 算法）
def estimate_rigid_transform(local_points: np.ndarray, global_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # local_points: (N, 3), global_points: (N, 3)
    n = local_points.shape[0]
    if n < 3:
        raise ValueError("At least 3 points are required for rigid transform estimation.")
    
    # 中心化
    local_centroid = np.mean(local_points, axis=0)
    global_centroid = np.mean(global_points, axis=0)
    local_centered = local_points - local_centroid
    global_centered = global_points - global_centroid
    
    # 计算协方差矩阵
    H = local_centered.T @ global_centered
    U, _, Vt = np.linalg.svd(H)
    
    # 计算旋转矩阵
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[:, -1] *= -1
        R = Vt.T @ U.T
    
    # 计算平移向量
    t = global_centroid - R 
    
    return R, t

# 图匹配：返回匹配的节点ID和对应点对
def match_subgraph(map_graph: nx.Graph, obs_graph: nx.Graph, label_weight: float = 0.5) -> Tuple[int, float, List[Tuple[int, int]]]:
    best_match_id = None
    best_score = float('inf')
    best_matches = []
    
    for node_id in map_graph.nodes:
        subgraph_nodes = [node_id] + list(map_graph.neighbors(node_id))
        map_subgraph = map_graph.subgraph(subgraph_nodes).copy()
        
        if nx.is_isomorphic(map_subgraph, obs_graph, node_match=lambda n1, n2: n1['label'] == n2['label']):
            score = 0.0
            node_matches = []
            for n1, n2 in zip(map_subgraph.nodes(data=True), obs_graph.nodes(data=True)):
                label_score = 0.0 if n1[1]['label'] == n2[1]['label'] else 1.0
                pos_dist = distance.euclidean(n1[1]['position'], n2[1]['position'])
                desc_dist = distance.cosine(n1[1]['descriptor'], n2[1]['descriptor'])
                score += label_weight * label_score + (1 - label_weight) * (pos_dist + desc_dist)
                node_matches.append((n1[0], n2[0]))  # 记录匹配的节点对
            
            if score < best_score:
                best_score = score
                best_match_id = node_id
                best_matches = node_matches
    
    return best_match_id, best_score, best_matches

# 重匹配主函数：返回机器人位姿
def relocalize(map_graph: nx.Graph, observed_objects: List[Dict], score_threshold: float = 5.0) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    obs_graph = generate_observation_subgraph(observed_objects)
    best_match_id, best_score, node_matches = match_subgraph(map_graph, obs_graph)
    
    if best_score > score_threshold or not node_matches:
        return None, None, None, None
    
    # 提取匹配点对的位置
    local_points = []
    global_points = []
    for map_id, obs_id in node_matches:
        local_points.append(observed_objects[obs_id]['position'])
        global_points.append(map_graph.nodes[map_id]['position'])
    
    local_points = np.array(local_points)
    global_points = np.array(global_points)
    
    # 计算刚体变换
    R, t = estimate_rigid_transform(local_points, global_points)
    
    # 机器人位置：假设相机在局部坐标系原点，位置为 t
    robot_position = t
    # 机器人朝向：从旋转矩阵提取欧拉角或四元数
    rotation = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
    
    return best_match_id, robot_position, rotation, global_points[0]  # 返回匹配节点ID、机器人位置、朝向和参考节点位置

# 示例用法
if __name__ == "__main__":
    # 模拟地图中的语义节点
    map_nodes = [
        SemanticNode(0, "table", np.array([0.0, 0.0, 0.0]), np.random.rand(128)),
        SemanticNode(1, "chair", np.array([1.0, 0.0, 0.0]), np.random.rand(128)),
        SemanticNode(2, "chair", np.array([0.0, 1.0, 0.0]), np.random.rand(128)),
        SemanticNode(3, "lamp", np.array([2.0, 2.0, 0.0]), np.random.rand(128))
    ]
    map_graph = build_semantic_graph(map_nodes)
    
    # 模拟当前观测到的物体
    observed_objects = [
        {'label': 'table', 'position': [0.1, 0.1, 0.0], 'descriptor': np.random.rand(128)},
        {'label': 'chair', 'position': [1.1, 0.1, 0.0], 'descriptor': np.random.rand(128)},
        {'label': 'chair', 'position': [0.1, 1.1, 0.0], 'descriptor': np.random.rand(128)}
    ]
    
    # 进行重匹配
    matched_id, robot_position, robot_rotation, ref_position = relocalize(map_graph, observed_objects)
    if matched_id is not None:
        print(f"Matched node ID: {matched_id}")
        print(f"Robot position (global): {robot_position}")
        print(f"Robot rotation (Euler angles, degrees): {robot_rotation}")
        print(f"Reference node position: {ref_position}")
    else:
        print("No valid match found.")


