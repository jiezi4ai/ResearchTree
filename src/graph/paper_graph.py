import networkx as nx
from typing import List, Dict, Union, List, Set, Tuple, Hashable # 用于类型提示

NodeType = Hashable # 节点类型通常是可哈希的

class PaperGraph:
    def __init__(self, name):
        self.graph = nx.MultiDiGraph(name=name)

    def add_graph_nodes(self, nodes_json: List[Dict]|Dict):
        """add paper node to graph
        Args:
            nodes_json (List[Dict] or Dict): original node json processed in format like:
                dct = {
                    "type": "node",
                    "id": ,
                    "labels": ["Paper"],
                    "properties": {"key":value}
                    }
        """
        if type(nodes_json) == dict:
            nodes_json = [nodes_json]

        nx_nodes_info = []
        for item in nodes_json:
            id = item['id']
            properties = item['properties']
            properties['nodeType'] = item['labels'][0]
            # be aware that node shall take the form like (1, dict(size=11)) for networkX
            nx_nodes_info.append((id, properties))  
        
        self.graph.add_nodes_from(nx_nodes_info)


    def add_graph_edges(self, edges_json: List[Dict]|Dict):
        """add paper node to graph
        Args:
            edges_json (List[Dict] or Dict): original relationship json processed in format like:
                dct = {
                        "type": "relationship",
                        "relationshipType": "WRITES",
                        "startNodeId": ,
                        "endNodeId": ,
                        "properties": {'authorOrder': author_order, 'coauthors': coauthors}
                        }
        """
        if type(edges_json) == dict:
            edges_json = [edges_json]

        nx_edges_info = []
        for item in edges_json:
            source_id = item['startNodeId']
            target_id = item['endNodeId']
            properties = item['properties']
            properties['relationshipType'] = item['relationshipType']
            # be aware that relationship shall take the form like (4, 5, dict(route=282)) for networkX
            nx_edges_info.append((source_id, target_id, properties))  
        
        self.graph.add_edges_from(nx_edges_info)
    

    def update_node_property(self, node_id, kv_dict):
        """update node properties in graph
        Args:
            node_id: unique node identifier
            kv_dict: information to update
        """
        assert node_id in self.graph.nodes

        for key in kv_dict.keys():
            value = kv_dict[key]
            self.graph.nodes[node_id][key] = value


    def update_edge_property(self, source_id, target_id, kv_dict):
        """update edge properties in graph
        Args:
            source_id, target_id: unique edge identifier
            kv_dict: information to update
        """
        assert (source_id, target_id) in self.graph.edges

        for key in kv_dict.keys():
            value = kv_dict[key]
            self.graph.edges[source_id, target_id][key] = value


    def find_wcc_subgraphs_for_nodes(
        self,
        target_nodes: Union[NodeType, List[NodeType], Set[NodeType], Tuple[NodeType]]
    ) -> List[nx.MultiDiGraph]:
        """查找包含一个或多个指定节点的弱连通分量对应的子图。
        Args:
            graph: NetworkX MultiDiGraph 图对象。
            target_nodes: 一个节点 ID，或一个包含节点 ID 的列表、集合或元组。
        Returns:
            一个包含所有找到的弱连通分量子图 (作为独立的 MultiDiGraph 副本) 的列表。
            如果目标节点不在图中或找不到对应的连通分量，则返回空列表。
            注意：如果多个目标节点在同一个连通分量中，该分量的子图只会被返回一次。
        """
        # 1. 标准化输入为集合
        if isinstance(target_nodes, (list, set, tuple)):
            target_nodes_set = set(target_nodes)
        else:
            # 假设是单个节点 ID
            target_nodes_set = {target_nodes}

        # 2. 检查所有目标节点是否存在于图中
        missing_nodes = target_nodes_set - set(self.graph.nodes())
        if missing_nodes:
            print(f"警告：以下目标节点不在图中，将被忽略: {missing_nodes}")
            target_nodes_set -= missing_nodes # 移除不存在的节点

        if not target_nodes_set:
            print("错误：没有有效的目标节点可供查找。")
            return []

        # 3. 查找并收集包含任何目标节点的弱连通分量
        found_subgraphs = []
        found_components_nodes = set() # 用于跟踪已添加的分量的节点集，避免重复

        for component_nodes in nx.weakly_connected_components(self.graph):
            component_set = set(component_nodes)
            # 4. 检查当前分量是否包含任何目标节点 (使用集合交集)
            if not target_nodes_set.isdisjoint(component_set): # 如果交集非空
                # 检查这个分量是否已经添加过 (基于其节点集合)
                # frozenset 是可哈希的，可以放入集合中
                component_frozenset = frozenset(component_set)
                if component_frozenset not in found_components_nodes:
                    # 5. 提取子图并添加到结果列表
                    subgraph = self.graph.subgraph(component_nodes).copy()
                    found_subgraphs.append(subgraph)
                    found_components_nodes.add(component_frozenset)

                    # Optional: 如果我们确定一个目标节点只能属于一个WCC,
                    # 可以在这里从 target_nodes_set 中移除 component_set 里的目标节点
                    # 以可能稍微提高后续迭代的效率，但这通常不是必需的
                    # target_nodes_set -= component_set

        return found_subgraphs


    def cal_node_significance(self):
        # networkx.pagerank(G, alpha=0.85, personalization=None, max_iter=100, tol=1e-06, nstart=None, weight='weight', dangling=None)
        """
        计算 NetworkX MultiDiGraph 的 PageRank，不使用内置函数。

        参数：
        G: 一个 NetworkX MultiDiGraph。
        alpha: 阻尼系数，默认值为 0.85。
        max_iter: 最大迭代次数，默认值为 100。
        tol: 收敛容差，默认值为 1e-06。
        weight: 用于表示边权重的边属性的键，默认值为 'weight'。
        personalization: 可选的个性化向量，字典形式。
        dangling: 可选的悬挂节点处理字典。

        返回：
        一个字典，包含每个节点的 PageRank 值。
        """
        if not G.is_directed():
            raise nx.NetworkXError("PageRank algorithm not defined for undirected graphs.")

        N = len(G)
        if N == 0:
            return {}

        if personalization is None:
            p = dict.fromkeys(G, 1.0 / N)
        else:
            s = sum(personalization.values())
            if s <= 0:
                raise ZeroDivisionError("Sum of personalization values must be positive.")
            p = {k: v / s for k, v in personalization.items()}

        if nstart is None:
            x = dict.fromkeys(G, 1.0 / N)
        else:
            s = sum(nstart.values())
            if s <= 0:
                raise ZeroDivisionError("Sum of starting values must be positive.")
            x = {k: v / s for k, v in nstart.items()}

        # 构建节点的出度权重字典
        out_degree_weights = {}
        for node in G:
            total_weight = 0
            for _, _, data in G.out_edges(node, data=True):
                total_weight += data.get(weight, 1)
            out_degree_weights[node] = total_weight

        # 找出悬挂节点
        dangling_nodes = [node for node in G if out_degree_weights[node] == 0]

        # 处理悬挂节点的目标
        if dangling is None:
            dangling_weights = p
        else:
            s = sum(dangling.values())
            if s <= 0:
                raise ZeroDivisionError("Sum of dangling node weights must be positive.")
            dangling_weights = {k: v / s for k, v in dangling.items()}

        for _ in range(max_iter):
            xlast = x
            x = dict.fromkeys(xlast, 0)
            danglesum = alpha * sum(xlast[n] for n in dangling_nodes)

            for node in G:
                for predecessor, _, data in G.in_edges(node, data=True):
                    edge_weight = data.get(weight, 1)
                    out_weight = out_degree_weights[predecessor]
                    if out_weight > 0:
                        x[node] += alpha * xlast[predecessor] * (edge_weight / out_weight)

            for node in G:
                x[node] += (1 - alpha) * p[node] + danglesum * dangling_weights.get(node, 0)

            err = sum([abs(x[n] - xlast[n]) for n in x])
            if err < N * tol:
                return x

        raise nx.PowerIterationFailedConvergence(f"PageRank failed to converge in {max_iter} iterations.")
