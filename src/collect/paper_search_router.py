
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Optional, Union, Any, Set, Tuple

import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from graph.paper_graph import PaperGraph
from collect.paper_similarity_calculation import PaperSim


class PaperRouter:
    def __init__(
        self,
        nodes_json: Optional[List[Dict]] = None,
        edges_json: Optional[List[Dict]] = None,
        paper_sim: Optional[PaperSim] = None, 
        
        embed_api_key: Optional[str] = None,
        embed_model_name: Optional[str] = None
        ):
        self.nodes_json = nodes_json
        self.edges_json = edges_json
        
        # initiate semantic scholar instances
        if sim_cal and isinstance(sim_cal, PaperSim):
            self.sim_cal = sim_cal  
        else:
            self.sim_cal = PaperSim(
                embed_api_key = embed_api_key,
                embed_model_name = embed_model_name
            )
    

    async def paper_sim_score(self, nodes_json):
        """calculate paper-paper similarity"""
        # valid paper with abstracts
        paper_json_w_abstract = [node for node in nodes_json 
                                if node['labels'] == ['Paper'] 
                                and node['properties'].get('title') is not None and node['properties'].get('abstract') is not None]
        paper_dois_w_abstract = [node['id'] for node in paper_json_w_abstract]

        # calculate paper nodes similarity
        semantic_similar_pool = await self.sim_cal.cal_embed_and_similarity(
            paper_nodes_json = paper_json_w_abstract,
            paper_dois_1 = paper_dois_w_abstract, 
            paper_dois_2 = paper_dois_w_abstract,
            similarity_threshold = 0.7,
            )
        return semantic_similar_pool


    async def topic_sim_score(self, nodes_json):
        """calculate paper-topic similarity"""
        # valid paper with abstracts
        paper_json_w_abstract = [node for node in nodes_json 
                                if node['labels'] == ['Paper'] 
                                and node['properties'].get('title') is not None and node['properties'].get('abstract') is not None]
        paper_dois_w_abstract = [node['id'] for node in paper_json_w_abstract]

        # calculate paper nodes similarity
        semantic_similar_pool = await self.sim_cal.cal_embed_and_similarity(
            paper_nodes_json = paper_json_w_abstract,
            paper_dois_1 = paper_dois_w_abstract, 
            paper_dois_2 = paper_dois_w_abstract,
            similarity_threshold = 0.7,
            )
        return semantic_similar_pool

    def paper_significance_scorer(
            self,
            paper_id,
            paper_graph
            ):
        """identify paper significance"""
        paper_node = paper_graph.nodes[paper_id]

        # basic info
        pub_dt = paper_node.get('publicationDate')
        tot_cit_cnt = paper_node.get('citationCount')
        tot_sig_cit_cnt = paper_node.get('influentialCitationCount')
        tot_ref_cnt = paper_node.get('referenceCount')
        
        # generated info
        tm_to_dt = relativedelta(datetime.now().date(), datetime.strptime(pub_dt, '%Y-%m-%d').date())  # publish time to date
        mth_to_dt = tm_to_dt.years * 12 + tm_to_dt.months
        mth_avg_cit_cnt = tot_cit_cnt / mth_to_dt

        # author info:  author -> WRITES -> this paper
        predecessors = paper_graph.predecessors(paper_id)
        for nid in predecessors:
            node_info = paper_graph.nodes[nid]
            node_type = node_info.get('nodeType')
            if node_type == 'Author':
                tot_paper_cnt = node_info.get('paperCount')
                tot_citation_cnt = node_info.get('citationCount')
                h_index = node_info.get('hIndex')

        # local citation info: other paper -> CITES -> this paper
        predecessors = paper_graph.predecessors(paper_id)
        for nid in predecessors:
            node_info = paper_graph.nodes[nid]
            node_type = node_info.get('nodeType')
            if node_type == 'Paper':
                tot_cit_cnt = node_info.get('citationCount')
                tot_sig_cit_cnt = node_info.get('influentialCitationCount')

        # local reference info: this paper -> CITES -> other paper
        successors = paper_graph.successors(paper_id)
        for nid in successors:
            node_info = paper_graph.nodes[nid]
            node_type = node_info.get('nodeType')
            if node_type == 'Paper':
                tot_cit_cnt = node_info.get('citationCount')
                tot_sig_cit_cnt = node_info.get('influentialCitationCount')


        pass


    def paper_filter(
            self,
            nodes_json,
            edges_json,
            core_paper_ids,
            max_hop: Optional[int] = 2
            ):
        # filter similar papers (for topic / paper search, recommendation)
        final_paper_ids = set()

        # add papers similar to core paper ids
        for item in edges_json:
            relationship = item.get('relationshipType')
            if relationship == 'SIMILAR_TO':
                start_id = item['startNodeId']
                end_id = item['endNodeId']
                if start_id in core_paper_ids and end_id not in core_paper_ids:
                    final_paper_ids.add(end_id)
                elif start_id not in core_paper_ids and end_id in core_paper_ids:
                    final_paper_ids.add(start_id)

        # add papers within citation chain of core paper ids
        for item in edges_json:
            relationship = item.get('relationshipType')
            if relationship == 'CITES':
                start_id = item['startNodeId']
                end_id = item['endNodeId']
                if start_id in core_paper_ids and end_id not in core_paper_ids:
                    final_paper_ids.add(end_id)
                elif start_id not in core_paper_ids and end_id in core_paper_ids:
                    final_paper_ids.add(start_id)

        # filter papers belong to specific topic
        citation_count_ref = {x['id']:x.get('properties', {}).get('citationCount', 0) 
                              for x in nodes_json if x.get('labels')==['Paper']}
        for item in edges_json:
            relationship = item.get('relationshipType')
            if edge_data.get('relationshipType') == 'DISCUSS':
                start_id = item['startNodeId']  # paper id
                if start_id not in core_paper_ids:
                    gloabl_citation = citation_count_ref.get(start_id, 0)
                    if gloabl_citation > 10 and u in hop_1_sim_paper_ids:
                        hop_1_topic_paper_ids.append(u)

        # recommendation, author papers
        final_paper_ids.update(set(list(core_paper_ids) + hop_1_sim_paper_ids + hop_1_citation_paper_ids))
        
        # for two-hop papers
        if max_hop > 1:
            hop_2_ids = []
            for u, v, edge_data in paper_graph.edges(data=True):
                if edge_data.get('relationshipType') == 'SIMILAR_TO' and edge_data.get('weight') > 0.7:
                    if u in final_paper_ids and v not in final_paper_ids:
                        hop_2_ids.append(v)
                    elif u not in final_paper_ids and v in final_paper_ids:
                        hop_2_ids.append(u)
            final_paper_ids.update(hop_2_ids)

        return final_paper_ids
        

    def cross_ref_discover(self):
        pass

    def key_author_discover(self):
        pass

    def paper_graph_pruning(
            self, 
            nodes_json, 
            edges_json,
            core_paper_ids
            ):
        # paper graph before pruning
        G_pre = PaperGraph(name='Paper Graph Before Pruning')
        G_pre.add_graph_nodes(nodes_json)
        G_pre.add_graph_edges(edges_json)
        # g_stat_pre = get_graph_stats(G_post)

        # pruning by connectivity - requiring core paper ids must be with graph
        sub_graphs = G_pre.find_wcc_subgraphs(target_nodes=core_paper_ids)
        if sub_graphs is not None and len(sub_graphs) > 0:
            G_post  = sub_graphs[0]
            # g_stat_post = get_graph_stats(G_post)  # get stats after prunning
        else:
            G_post = G_pre

        
        pass






if iteration == 1:  # take paper searched in initial round as core paper
    # identify core papers and core authors nodes
    core_paper_ids = set(node['id'] for node in ps.nodes_json if node['labels'] == ['Paper'])
    core_author_ids = set(node['id'] for node in ps.nodes_json if node['labels'] == ['Author'])
    print(len(core_paper_ids), len(core_author_ids))
    ps.explored_nodes['paper'].update(core_paper_ids)

    # for reference and citings
    ref_ids_to_search = [pid for pid in core_paper_ids if pid not in ps.explored_nodes['reference']]
    cit_ids_to_search = [pid for pid in core_paper_ids if pid not in ps.explored_nodes['citing']]

    # for recommendations 
    pos_ids_to_search, neg_ids_to_search = None, None
    if len(ps.explored_nodes['recommendation']) == 0:
        if len(core_paper_ids) > 3:
            pos_ids_to_search = list(core_paper_ids)
            neg_ids_to_search = []

    # for topics generation
    core_paper_json = [x for x in ps.nodes_json if x['id'] in core_paper_ids]
    if len(ps.explored_nodes['topic']) < 4:  # explored topic less than 4, generate new topics
        await ps.topic_generation(
            paper_json = core_paper_json,
            llm_api_key = llm_api_key,
            llm_model_name = llm_model_name,
            )

    # identify unexplored topics
    # covert topic data to k-v format
    topic_pids = {}

    for item in ps.data_pool['topic']:
        topic = item['topic']
        paper_id = item['paperId']
        
        if topic not in topic_pids:
            topic_pids[topic] = []
            
        topic_pids[topic].append(paper_id)

    # identify topics with insufficient papers
    topics_to_search = []
    for topic, pids in topic_pids.items():
        if len(pids) < 10:
            topics_to_search.append(topic)
    print(topics_to_search)

else:  # take crossref papers as core papers
