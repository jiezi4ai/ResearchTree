import networkx as nx

import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from graph.graph_viz import GraphViz

def test():
    # 已有图结构
    G = nx.MultiDiGraph()

    # 添加节点（已省略，保持与问题中一致）
    # --- Create a Sample MultiDiGraph ---
    G = nx.MultiDiGraph()

    # Add nodes with types and attributes
    G.add_node("Paper1", nodeType='Paper', title='Intro to Graphs', year=2021, vizLabel="P1:Intro") # Add some base attributes
    G.add_node("Paper2", nodeType='Paper', title='Advanced Networks', year=2022, vizLabel="P2:Adv")
    G.add_node("Paper3", nodeType='Paper', title='Visualization Techniques', year=2023, vizLabel="P3:Viz")
    G.add_node("Author1", nodeType='Author', name='Alice', affiliation='Inst A', vizLabel="A1:Alice")
    G.add_node("Author2", nodeType='Author', name='Bob', affiliation='Inst B', vizLabel="A2:Bob")
    G.add_node("Venue1", nodeType='Venue', name='Conf X', vizLabel="V1:ConfX")
    G.add_node("Journal1", nodeType='Journal', name='Journal Y', vizLabel="J1:JY")
    G.add_node("MissingTypeNode", vizLabel="M?") # Node without 'nodeType'

    # Add edges with types and attributes
    G.add_edge("Author1", "Paper1", relationshipType='WRITES', weight=0.8, vizLabel="A1 writes P1")
    G.add_edge("Author1", "Paper2", relationshipType='WRITES', weight=0.9, vizLabel="A1 writes P2")
    G.add_edge("Author2", "Paper1", relationshipType='WRITES', weight=0.7, vizLabel="A2 writes P1")
    G.add_edge("Author2", "Paper3", relationshipType='WRITES', weight=0.8, vizLabel="A2 writes P3")
    G.add_edge("Paper2", "Paper1", relationshipType='CITES', weight=0.5, vizLabel="P2 cites P1") 
    G.add_edge("Paper3", "Paper1", relationshipType='CITES', weight=0.6, vizLabel="P3 cites P1")
    G.add_edge("Paper3", "Paper2", relationshipType='CITES', weight=0.4, vizLabel="P3 cites P2")
    G.add_edge("Paper1", "Venue1", relationshipType='RELEASES_IN', weight=0.2, vizLabel="P1 in V1")
    G.add_edge("Paper2", "Journal1", relationshipType='PRINTS_ON', weight=0.3, vizLabel="P2 on J1")
    G.add_edge("Paper3", "Venue1", relationshipType='RELEASES_IN', weight=0.2, vizLabel="P3 in V1")
    # Add an edge without a relationshipType
    G.add_edge("Author1", "Author2", vizLabel="A1 -> A2 (Unknown)")
    # Add a parallel edge
    G.add_edge("Author1", "Paper1", key="review", relationshipType='REVIEWS', weight=0.1, vizLabel="A1 reviews P1")

    viz = GraphViz(G, 'test')
    viz.preprocessing()
    viz.visulization()


if __name__ == "__main__":
    test()