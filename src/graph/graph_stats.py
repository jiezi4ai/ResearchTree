import copy
import numpy as np
from collections import Counter
from typing import List, Dict, Optional, Union, Tuple, Literal # Added Tuple

from graph.paper_graph import PaperGraph

def get_graph_stats(graph):
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


def get_paper_stats(graph, seed_paper_dois):
    """get paper statistic in paper graph"""
    papers_stats = []
    for nid, node_data in graph.nodes(data=True):
        if node_data.get('nodeType') == 'Paper':
            # paper infos
            title = graph.nodes[nid].get('title')
            if_seed = True if nid in seed_paper_dois else False  # exclude seed papers
            overall_cite_cnt = node_data.get('citationCount')
            overall_inf_cite_cnt = node_data.get('influentialCitationCount')
            overall_ref_cnt = node_data.get('influentialCitationCount')

            # for in edges
            in_edges_info = graph.in_edges(nid, data=True)
            local_citation_cnt = 0  # local paper graph cites papers (other cites this one)
            sim_cnt_1 = 0  # local paper graph similar papers to this one
            max_sim_to_seed_1 = -1  # max similarity of this paper to seed papers
            for u, _, edge_data in in_edges_info:
                if edge_data.get('relationshipType') == 'CITES':
                    local_citation_cnt += 1
                elif edge_data.get('relationshipType') == 'SIMILAR_TO':
                    sim_cnt_1 += 1
                    if u in seed_paper_dois:
                        if edge_data.get('weight') > max_sim_to_seed_1:
                            max_sim_to_seed_1 = edge_data.get('weight')

            # for out edges
            out_edges_info = graph.out_edges(nid, data=True)
            local_ref_cnt = 0  # local paper graph cites papers (other cites this one)
            sim_cnt_2 = 0  # local paper graph similar papers to this one
            max_sim_to_seed_2 = -1  # max similarity of this paper to seed papers
            for _, v, edge_data in out_edges_info:
                if edge_data.get('relationshipType') == 'CITES':
                    local_ref_cnt += 1
                elif edge_data.get('relationshipType') == 'SIMILAR_TO':
                    sim_cnt_2 += 1
                    if v in seed_paper_dois:
                        if edge_data.get('weight') > max_sim_to_seed_2:
                            max_sim_to_seed_2 = edge_data.get('weight')

            # author infors
            author_ids_lst = [x['authorId'] for x in node_data.get('authors', []) if x.get('authorId') is not None]
            tot_author_cnt = len(author_ids_lst)

            # get author order and h-index
            h_index_lst, author_order_lst = [], []
            for idx, aid in enumerate(author_ids_lst):
                author_order = idx + 1
                h_index = graph.nodes[aid].get('hIndex')
                if h_index is not None:
                    h_index_lst.append(h_index)
                    author_order_lst.append(author_order)

            if len(h_index_lst) > 0:
                avg_h_index = np.average(h_index_lst)
                weight_h_index = sum([x / y for x, y in zip(h_index_lst, author_order_lst)]) / len(h_index_lst)
            else:
                avg_h_index = None
                weight_h_index = None

            paper_stats = {"doi":nid, "title":title, "if_seed": if_seed,
                           "local_citation_cnt":local_citation_cnt, "local_reference_cnt": local_ref_cnt, 
                           "local_similarity_cnt":sim_cnt_1+sim_cnt_2, "max_sim_to_seed":max(max_sim_to_seed_1, max_sim_to_seed_2),
                           "global_citaion_cnt":overall_cite_cnt, "influencial_citation_cnt":overall_inf_cite_cnt, "global_refence_cnt": overall_ref_cnt,
                           "author_cnt":tot_author_cnt, "avg_h_index":avg_h_index, 'weighted_h_index':weight_h_index}
            papers_stats.append(paper_stats)
    return papers_stats


def get_author_stats(graph, seed_author_ids):
    """get author statistic in paper graph"""

    h_index_ref = {nid:node_data['hIndex'] for nid, node_data in graph.nodes(data=True) if node_data.get('nodeType') == 'Author' 
                   and node_data.get('hIndex') is not None}

    authors_stats = []
    for nid, node_data in graph.nodes(data=True):
        if node_data.get('nodeType') == 'Author':
            # properties
            author_name = node_data.get('name')
            h_index = node_data.get('hIndex')
            if_seed = True if nid in seed_author_ids else False  # exclude seed authors
            if_complete = node_data.get('is_complete', False)
            global_paper_cnt = node_data.get('paperCount')
            global_citation_cnt = node_data.get('citationCount')

            # local stats
            out_edges_info = graph.out_edges(nid, data=True)
            local_paper_cnt = sum([1 for _, _, data in out_edges_info if data.get('relationshipType') == 'WRITES'])
            # get coauthors
            coauthor_ids = []
            for u,v, edge_data in out_edges_info:
                if edge_data.get('relationshipType') == 'WRITES':
                    coauthors = edge_data.get('coauthors', [])
                    coauthor_ids.extend([x['authorId'] for x in coauthors if x.get('authorId') is not None])
            
            # get top coauthors
            coauthor_cnt = Counter(coauthor_ids)
            top_coauthors = coauthor_cnt.most_common()[0:5]  # rank order by descending

            # calculate top coauthor h-index
            coauthor_cnt = 0
            sum_coauthor_h_index = 0
            for idx, item in enumerate(top_coauthors):
                coauthor_id = item[0]
                coauthor_hindex = h_index_ref.get(coauthor_id)
                if coauthor_hindex is not None:
                    sum_coauthor_h_index += coauthor_hindex /(idx + 1)
                    coauthor_cnt += 1
            weighted_coauthor_h_index = sum_coauthor_h_index / coauthor_cnt if coauthor_cnt > 0 else None

            author_stat = {"author_id":nid, "author_name":author_name, "if_seed":if_seed, 'if_complete':if_complete, 
                           "h_index":h_index, "global_paper_cnt":global_paper_cnt, "global_citation_cnt":global_citation_cnt,
                           "local_paper_cnt":local_paper_cnt, 
                           "top_coauthors":top_coauthors, "weighted_coauthor_h_index": weighted_coauthor_h_index
                          }
            authors_stats.append(author_stat)
    return authors_stats
