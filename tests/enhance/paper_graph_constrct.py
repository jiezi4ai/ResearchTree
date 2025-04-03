import networkx as nx
from typing import List, Dict

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
            nx_nodes_info.append(id, properties)  
        
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
            nx_edges_info.append(source_id, target_id, properties)  
        
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