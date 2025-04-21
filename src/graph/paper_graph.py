import networkx as nx
import community as community_louvain  # pip install python-louvain  https://github.com/taynaud/python-louvain
from networkx.algorithms.community import label_propagation_communities
from typing import List, Dict, Union, List, Set, Tuple, Hashable, Literal, Optional

NodeType = Hashable # 节点类型通常是可哈希的

class PaperGraph(nx.MultiDiGraph):
    def __init__(self, name:Optional[str]='Paper Graph'):
        super().__init__(name=name)

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
        
        self.add_nodes_from(nx_nodes_info)


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
        
        self.add_edges_from(nx_edges_info)
    

    def update_node_property(self, node_id, kv_dict):
        """update node properties in graph
        Args:
            node_id: unique node identifier
            kv_dict: information to update
        """
        assert node_id in self.nodes

        for key in kv_dict.keys():
            value = kv_dict[key]
            self.nodes[node_id][key] = value


    def update_edge_property(self, source_id, target_id, kv_dict):
        """update edge properties in graph
        Args:
            source_id, target_id: unique edge identifier
            kv_dict: information to update
        """
        assert (source_id, target_id) in self.edges

        for key in kv_dict.keys():
            value = kv_dict[key]
            self.edges[source_id, target_id][key] = value


    def find_common_paths_between_pairs(self, nodes):
        """
        找到用户输入的每两个节点之间的所有简单路径。
        """
        graph = self.to_undirected()
        if not nodes or len(nodes) < 2:
            return {}

        all_paths = {}
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                source_node = nodes[i]
                target_node = nodes[j]
                try:
                    paths = list(nx.all_simple_paths(graph, source=source_node, target=target_node))
                    if paths:
                        all_paths[(source_node, target_node)] = paths
                    else:
                        print(f"节点 {source_node} 和 {target_node} 之间没有简单路径。")
                except nx.NodeNotFound as e:
                    print(f"节点 {e} 不在图中。")
                    return None
        return all_paths


    def find_shortest_paths_between_pairs(self, nodes):
        """
        找到用户输入的每两个节点之间的最短路径。
        """
        graph = self.to_undirected()
        if not nodes or len(nodes) < 2:
            return {}

        shortest_paths = {}
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                source_node = nodes[i]
                target_node = nodes[j]
                try:
                    path = nx.shortest_path(graph, source=source_node, target=target_node)
                    shortest_paths[(source_node, target_node)] = path
                except nx.NetworkXNoPath:
                    print(f"节点 {source_node} 和 {target_node} 之间没有路径。")
                except nx.NodeNotFound as e:
                    print(f"节点 {e} 不在图中。")
                    return None
        return shortest_paths


    def find_paths_connecting_all_nodes(self, nodes):
        """
        找到连接所有指定节点的最短路径的组合。
        """
        graph = self.to_undirected()

        if not nodes or len(nodes) < 2:
            return

        all_paths = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                try:
                    shortest_path = nx.shortest_path(graph, source=nodes[i], target=nodes[j])
                    all_paths.append(shortest_path)
                except nx.NetworkXNoPath:
                    print(f"节点 {nodes[i]} 和 {nodes[j]} 之间没有路径。")
                    return None

        # 这里可以进一步处理 all_paths 来合并或分析连接所有节点的路径
        # 例如，可以提取所有路径中的边，构建一个包含这些边的子图。
        edges = set()
        for path in all_paths:
            for i in range(len(path) - 1):
                u, v = sorted((path[i], path[i+1])) # 考虑无向图，对节点排序
                edges.add((u, v))

        connecting_subgraph = nx.Graph(list(edges)) # 创建包含这些边的子图
        nodes_in_subgraph = set(connecting_subgraph.nodes())
        if all(node in nodes_in_subgraph for node in nodes):
            return connecting_subgraph
        else:
            print("无法找到直接连接所有指定节点的路径组合。")
            return None


    def find_wcc_subgraphs(
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
        missing_nodes = target_nodes_set - set(self.nodes())
        if missing_nodes:
            print(f"警告：以下目标节点不在图中，将被忽略: {missing_nodes}")
            target_nodes_set -= missing_nodes # 移除不存在的节点

        if not target_nodes_set:
            print("错误：没有有效的目标节点可供查找。")
            return []

        # 3. 查找并收集包含任何目标节点的弱连通分量
        found_subgraphs = []
        found_components_nodes = set() # 用于跟踪已添加的分量的节点集，避免重复

        for component_nodes in nx.weakly_connected_components(self):
            component_set = set(component_nodes)
            # 4. 检查当前分量是否包含任何目标节点 (使用集合交集)
            if not target_nodes_set.isdisjoint(component_set): # 如果交集非空
                # 检查这个分量是否已经添加过 (基于其节点集合)
                # frozenset 是可哈希的，可以放入集合中
                component_frozenset = frozenset(component_set)
                if component_frozenset not in found_components_nodes:
                    # 5. 提取子图并添加到结果列表
                    subgraph = self.subgraph(component_nodes).copy()
                    found_subgraphs.append(subgraph)
                    found_components_nodes.add(component_frozenset)

                    # Optional: 如果我们确定一个目标节点只能属于一个WCC,
                    # 可以在这里从 target_nodes_set 中移除 component_set 里的目标节点
                    # 以可能稍微提高后续迭代的效率，但这通常不是必需的
                    # target_nodes_set -= component_set

        return found_subgraphs
    

    def detect_louvain_community(
            self, 
            community_type: Literal['louvain',  # louvian community detection algorithm
                                    'lpa',  # label propagation community detection algorithm
                                    ] = 'louvain'
            ):
        graph = self.to_undirected()
        if community_type == 'Louvain':  # 使用 Louvain 算法发现社群
            communities = community_louvain.best_partition(graph)
        elif community_type == 'lpa': # 使用标签传播算法发现社群
            communities = list(label_propagation_communities(graph))
        return communities

        # # 查找特定节点所属的社群 (需要遍历社群)
        # specific_nodes = seed_dois
        # for node in specific_nodes:
        #     found = False
        #     for key, value in partition.items():
        #         if node in key:
        #             print(f"节点 {node} 属于Louvain发现的社群 {value}: {key}")
        #             found = True
        #             break
        #     if not found:
        #         print(f"节点 {node} 不在任何已发现的社群中")


    def cal_node_centrality(
            self, 
            type_of_centrality:Optional[Literal['in',  # in degree centrality
                                                'out',   # out degree centrality
                                                'between',  # betweenness centrality 
                                                'closeness'  # closeness centrality
                                                ]] = 'in'
            ):
        """Calculate node centrality and sort in descending order
        """
        if type_of_centrality == 'in':  # 计算入度中心性
            centrality = nx.in_degree_centrality(self)
        elif type_of_centrality == 'out':  # 计算出度中心性
            centrality = nx.out_degree_centrality(self)
        elif type_of_centrality == 'between':  # 计算介数中心性
            centrality = nx.betweenness_centrality(self)
        elif type_of_centrality == 'closeness':  # 计算紧密中心性
            centrality = nx.closeness_centrality(self)
        
        sorted_centrality = sorted(centrality.items(), key=lambda item: item[1], reverse=True)
        return sorted_centrality


    def assign_edge_weight_mannually(self):
        for u, v in self.edges():
            if self.edges[u, v]['relationshipType'] in ['CITES']:  # paper -cites - paper
                self.edges[u, v]['weight'] = 0.5
            elif self.edges[u, v]['relationshipType'] in ['DISCUSS']:  # paper - discuss - topic
                self.edges[u, v]['weight'] = 0.4
            elif self.edges[u, v]['relationshipType'] in ['WRITES']:  # author - writes - paper
                self.edges[u, v]['weight'] = 0.3 
            elif self.edges[u, v]['relationshipType'] in ['WORKS_IN']:  # author -works in - affliation
                self.edges[u, v]['weight'] = 0.2
            elif self.edges[u, v]['relationshipType'] in ['PRINTS_ON', 'RELEASES_IN']:  # paper - prints on / relaeses in - journal / veneue
                self.edges[u, v]['weight'] = 0.1
        return self


    def convert_to_homogeneous_graph(self, candit_edges):
        candit_ref = {(x['startNodeId'], x['endNodeId']):x['properties']['weight'] for x in candit_edges}
        filtered_nodes, filtered_edges = [], []

        # 过滤 ["Journal"] ["Venue"] ["Affiliation"] 节点
        for node_id in self.nodes():
            if self.nodes[node_id]['labels'][0] not in ['Journal', 'Venue', 'Affiliation']:
                filtered_nodes.append(node_id)

        # 过滤 WORKS_IN, PRINTS_ON, RELEASES_IN等类型的边
        for u, v in self.edges():
            if self.edges[u, v]['relationshipType'] not in ['WORKS_IN', 'PRINTS_ON', 'RELEASES_IN']: 
                filtered_edges.append((u, v))
        
        # 处理['Author']节点
        tmp_ref = []
        for node_id in self.nodes():
            if self.nodes[node_id]['labels'][0] in ['Author']:
                # first get author node id
                author_id = self.nodes[node_id]['id']
                # then get all sucessors for author node id
                author_id_successors = self.successors(author_id) 
                paper_id_publish_dt_ref = {id:self.nodes[node_id]['publicationDate']
                    for id in author_id_successors if self.nodes[node_id]['labels'][0] in ['Paper']}
                for paper_id_1, publish_dt_1 in paper_id_publish_dt_ref.items():
                    for paper_id_2, publish_dt_2 in paper_id_publish_dt_ref.items():
                        if paper_id_1 != paper_id_2 and publish_dt_1 <= publish_dt_2:
                            weight = candit_ref.get((paper_id_1, paper_id_2))
                            if weight > 0.5:
                                filtered_edges.append({
                                    "type": "relationship",
                                    "relationshipType": "COAUTHOR",
                                    "startNodeId": paper_id_1,
                                    "endNodeId": paper_id_2,
                                    "properties": {'weight': weight}
                                    })

        # 处理 ['Topic']节点
        tmp_ref = []
        for node_id in self.nodes():
            if self.nodes[node_id]['labels'][0] in ['Topic']:
                # first get author node id
                author_id = self.nodes[node_id]['id']
                # then get all sucessors for author node id
                author_id_successors = self.successors(author_id) 
                paper_id_publish_dt_ref = {id:self.nodes[node_id]['publicationDate']
                    for id in author_id_successors if self.nodes[node_id]['labels'][0] in ['Paper']}
                for paper_id_1, publish_dt_1 in paper_id_publish_dt_ref.items():
                    for paper_id_2, publish_dt_2 in paper_id_publish_dt_ref.items():
                        if paper_id_1 != paper_id_2 and publish_dt_1 <= publish_dt_2:
                            weight = candit_ref.get((paper_id_1, paper_id_2))
                            if weight > 0.5:
                                filtered_edges.append({
                                    "type": "relationship",
                                    "relationshipType": "COAUTHOR",
                                    "startNodeId": paper_id_1,
                                    "endNodeId": paper_id_2,
                                    "properties": {'weight': weight}
                                    })
                                
        # 仅保留边的起始和终结都是Papers节点
        # 仅保留Papers节点
        # 删除孤立的边和节点
        pass

    def cal_node_pagerank(self, G, alpha=0.85, personalization=None, max_iter=100, tol=1e-06, nstart=None, weight='weight', dangling=None):
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
