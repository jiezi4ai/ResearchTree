
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Optional, Union, Any, Set, Tuple

import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from graph.paper_graph import PaperGraph
from collect.paper_sim_calc import PaperSim


from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np


def graph_basic_stats(node_stats: Dict):
    """calculate key stats index for node stats"""
    node_stats_index = {}
    for key, values in node_stats.items():
        if key == 'id': continue  # skip id

        valid_values = [v for v in values if isinstance(v, (int, float))] # 只处理数值类型
        if valid_values:
            node_stats_index[key] = {
                'min': np.min(valid_values),
                'max': np.max(valid_values),
                'average': np.mean(valid_values),
                'median': np.median(valid_values),
                'quantile_25': np.percentile(valid_values, 25),
                'quantile_75': np.percentile(valid_values, 75)
            }
        else:
            node_stats_index[key] = {}  
    return node_stats_index


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
        if paper_sim and isinstance(paper_sim, PaperSim):
            self.sim_calc = paper_sim  
        else:
            self.sim_calc = PaperSim(
                embed_api_key = embed_api_key,
                embed_model_name = embed_model_name
            )


    ####################################################################################
    # similarity calculation and filter
    ####################################################################################
    async def score_paper2paper__sim(self, nodes_json):
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

    async def score_paper2topic_sim(self, nodes_json):
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


    ####################################################################################
    # statistics calculation and significant paper identify
    ####################################################################################
    def gen_nodes_stats(self, paper_graph):
        """calculate statistics for paper node"""
        # ---------- 1. Initiate stats  ------------
        # for paper stats
        paper_stats = {
            'id': [],
            'citationCount': [],
            'influentialCitationCount': [],
            'referenceCount': [],
            'monthlyCitationCount': [],
            'localCitationCount': [],
            'localReferenceCount': []
        }

        # for author stats
        author_stats = {
            'paperCount': [],
            'citationCount': [],
            'hIndex': [],
            'localPaperCount': []
        }

        # iterate paper node to calculate measurements
        for nid, node_data in paper_graph.nodes(data=True):
            # ---------- 2. Calculate paper stats  ------------
            if node_data.get('nodeType') == 'Paper':
                paper_stats['id'].append(nid)
                # global measurement
                pub_dt = node_data.get('publicationDate')
                paper_cit_cnt = node_data.get('citationCount')
                paper_stats['citationCount'].append(paper_cit_cnt)
                paper_stats['influentialCitationCount'].append(node_data.get('influentialCitationCount'))
                paper_stats['referenceCount'].append(node_data.get('referenceCount'))

                # generated measurement
                if pub_dt:
                    try:
                        tm_to_dt = relativedelta(datetime.now().date(), datetime.strptime(pub_dt, '%Y-%m-%d').date())
                        mth_to_dt = tm_to_dt.years * 12 + tm_to_dt.months
                        if paper_cit_cnt is not None and mth_to_dt > 0:
                            paper_mthly_cit_cnt = paper_cit_cnt / mth_to_dt
                            paper_stats['monthlyCitationCount'].append(paper_mthly_cit_cnt)
                    except ValueError:
                        paper_stats['monthlyCitationCount'].append(None)

                # local measurement
                predecessors = paper_graph.predecessors(nid)
                paper_loc_cit_cnt = sum([1 for x in predecessors if paper_graph.nodes[x].get('nodeType') == 'Paper'])
                paper_stats['localCitationCount'].append(paper_loc_cit_cnt)

                successors = paper_graph.successors(nid)
                paper_loc_ref_cnt = sum([1 for x in successors if paper_graph.nodes[x].get('nodeType') == 'Paper'])
                paper_stats['localReferenceCount'].append(paper_loc_ref_cnt)

            # ---------- 3. Calculate author stats  ------------
            elif node_data.get('nodeType') == 'Author':
                author_stats['id'].append(nid)
                # global measurement
                author_stats['paperCount'].append(node_data.get('paperCount'))
                author_stats['citationCount'].append(node_data.get('citationCount'))
                author_stats['hIndex'].append(node_data.get('hIndex'))

                # local measurement
                successors = paper_graph.successors(nid)
                author_loc_paper_cnt = sum([1 for x in successors if paper_graph.nodes[x].get('nodeType') == 'Paper'])
                author_stats['localPaperCount'].append(author_loc_paper_cnt)

        node_stats = {'paper_stats': paper_stats, 'author_stats': author_stats}

        # ---------- 4. generate stats analysis ------------
        paper_stats_analysis = graph_basic_stats(paper_stats)
        author_stats_analysis = graph_basic_stats(author_stats)
        stats_analysis = {'paper_stats_analysis': paper_stats_analysis, 
                          'author_stats_analysis': author_stats_analysis}

        return node_stats, stats_analysis

    def identify_paper_significant(
            self,
            paper_stats,
            ):
        """identify significant paper node"""
        paper_stats_df = pd.DataFrame(paper_stats)
        for index, row in paper_stats_df.iterrows():
            pid = row['id']
            # rule 1: global citation greater than or equal to 20
            if row['citationCount'] >= 20:
                significant_ind = 1
                info = 'citationCount'

            # rule 2: influential citation greater than or equal to 3
            elif row['influentialCitationCount'] >= 3:
                significant_ind = 1
                info = 'influentialCitationCount'

            # rule 3: monthly citation greater than or equal to 5
            elif row['monthlyCitationCount'] >= 5:
                significant_ind = 1
                info = 'monthlyCitationCount'

            # rule 4: local citation greater than or equal to 5
            elif row['localCitationCount'] >= 5:
                significant_ind = 1
                info = 'localCitationCount'

            row['significance'] = significant_ind
            row['sig_info'] = info

    def identify_author_significant(
            self,
            author_stats,
            ):
        """identify significant paper node"""
        author_stats_df = pd.DataFrame(author_stats)
        for index, row in author_stats_df.iterrows():
            aid = row['id']
            # rule 1: h-index greater than or equal to 10
            if row['hIndex'] >= 10:
                significant_ind = 1

            # rule 2: average paper ciation greater than or equal to 20
            elif row['paperCount'] / row['paperCount'] >= 20:
                significant_ind = 1

            # rule 4: local citation greater than or equal to 5
            elif row['localPaperCount'] >= 5:
                significant_ind = 1

            row['significance'] = significant_ind

    def identify_paper_significant_relative(self):
        pass

    def identify_paper_significant_lpa(self):
        pass

    ####################################################################################
    # statistics calculation and filter
    ####################################################################################
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
