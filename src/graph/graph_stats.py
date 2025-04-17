import copy
import numpy as np
from collections import Counter
from typing import List, Dict, Optional, Union, Tuple, Literal # Added Tuple

from paper_graph import PaperGraph

def basic_stats(graph):
    """basic stats for graph"""
    graph_stats = {}
    graph_stats['node_cnt'] = len(graph.nodes)
    graph_stats['edge_cnt'] = len(graph.edges)
    print(f"Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges.")

    # check node types
    node_types = [node_data.get('nodeType') for _, node_data in graph.nodes(data=True)]
    node_types_cnt = Counter(node_types)
    sorted_node_counts = node_types_cnt.most_common()  # rank order by descending
    graph_stats['node_type'] = sorted_node_counts  # format like [(node type, nodes count), ...]
    print(f"There are {len(sorted_node_counts)} node types in this graph, they are:\n{sorted_node_counts}")

    # check edge types
    edge_types = [d.get('relationshipType') for _, _, d in graph.edges(data=True)]
    edge_types_cnt = Counter(edge_types)
    sorted_egdes_counts = edge_types_cnt.most_common()  # rank order by descending
    graph_stats['edge_type'] = sorted_egdes_counts  # format like [(node type, nodes count), ...]
    print(f"There are {len(sorted_egdes_counts)} edge types in this graph, they are:\n{sorted_egdes_counts}")

    return graph_stats


def paper_stats(graph:PaperGraph, seed_paper_dois, order_by: Optional[Literal['reference', 'cited', 'similarity']]='cited'):
    """get paper statistic in paper graph"""

    basic_stat = []
    for nid, node_data in graph.nodes:
        if node_data.get('nodeType') == 'Paper':
            # for in edges
            in_edges_info = graph.in_edges(nid, data=True)
            cite_cnt = sum([1 for _, _, edge_data in in_edges_info if edge_data.get('relationshipType') == 'CITES'])
            sim_cnt_1 = sum([1 for _, _, edge_data in in_edges_info if edge_data.get('relationshipType') == 'SIMILAR_TO'])
            
            # for out edges
            out_edges_info = graph.out_edges(nid, data=True)
            cited_cnt = sum([1 for _, _, edge_data in out_edges_info if edge_data.get('relationshipType') == 'CITES'])
            sim_cnt_2 = sum([1 for _, _, edge_data in out_edges_info if edge_data.get('relationshipType') == 'SIMILAR_TO'])

            sim_cnt = sim_cnt_1 + sim_cnt_2
            basic_stat.append((n, cite_cnt, cited_cnt, sim_cnt))

    if order_by == 'cited':
        sorted_stat = sorted(basic_stat, key=lambda item: item[1], reverse=True)
    elif order_by == 'reference':
        sorted_stat = sorted(basic_stat, key=lambda item: item[2], reverse=True) 
    elif order_by == 'similarity':
        sorted_stat = sorted(basic_stat, key=lambda item: item[3], reverse=True) 

    papers_stats = []
    for item in sorted_stat:
        n = item[0]
        local_citation_cnt = item[1]
        local_ref_cnt = item[2]
        local_sim_cnt = item[3]

        # paper infos
        title = graph.nodes[n].get('title')
        in_seed = True if item[0] in seed_paper_dois else False
        overall_cite_cnt = graph.nodes[n].get('citationCount')
        overall_inf_cite_cnt = graph.nodes[n].get('influentialCitationCount')
        overall_ref_cnt = graph.nodes[n].get('influentialCitationCount')

        # author infors
        tot_author_cnt = sum([1 for u in graph.predecessors(n) if graph.nodes[u].get('nodeType') == 'Author'])
        h_index_lst, author_order_lst = [], []
        for u in graph.predecessors(n):
            if graph.nodes[u].get('nodeType') == 'Author':
                h_index = graph.nodes[u].get('hIndex')
                author_order = graph[u][n].get('authorOrder')
                if h_index:
                    h_index_lst.append(h_index)
                    author_order_lst.append(author_order)
                    
        avg_h_index = np.average(h_index_lst)
        weight_h_index = sum([x / y for x, y in zip(h_index_lst, author_order_lst)]) / len(h_index_lst)

        paper_stats = {"doi":n, "title":title, "if_seed": in_seed,
                      "local_citation_cnt":local_citation_cnt, "local_reference_cnt": local_ref_cnt, "local_similarity_cnt":local_sim_cnt,
                      "global_citaion_cnt":overall_cite_cnt, "influencial_citation_cnt":overall_inf_cite_cnt, "global_refence_cnt": overall_ref_cnt,
                      "author_cnt":tot_author_cnt, "avg_h_index":avg_h_index, 'weighted_h_index':weight_h_index}
        papers_stats.append(paper_stats)

        return papers_stats
    
def author_stats(graph, seed_author_ids, order_by:Optional[str]='write'):
    """get author statistic in paper graph"""
    basic_stat = []
    for n in graph.nodes:
        if graph.nodes[n].get('nodeType') == 'Author':
            out_edges_info = graph.out_edges(n, data=True)
            write_cnt = sum([1 for u, v, data in out_edges_info if data.get('relationshipType') == 'WRITES'])
            basic_stat.append((n, write_cnt))

    sorted_by_writes = sorted(basic_stat, key=lambda item: item[1], reverse=True)

    authors_stats = []
    for item in sorted_by_writes:
        aid = item[0]
        a_name = graph.nodes[aid].get('name')
        hIndex = graph.nodes[aid].get('hIndex')
        in_seed = True if aid in seed_author_ids else False
        global_paper_cnt = graph.nodes[aid].get('paperCount')
        global_citation_cnt = graph.nodes[aid].get('citationCount')
        
        author_stat = {"author_id":aid, "author_name":a_name, 
                       "write_cnt":item[1], "is_seed":in_seed,
                       "hIndex":hIndex, "global_paper_cnt":global_paper_cnt, 
                       "global_citation_cnt":global_citation_cnt, }
        authors_stats.append(author_stat)
    return authors_stats