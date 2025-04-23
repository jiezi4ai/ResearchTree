# paper_collection.py
# core function for paper collection

import copy
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Tuple, Literal

import os
import sys
import json

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from graph.paper_graph import PaperGraph
from graph.graph_stats import get_graph_stats, get_author_stats, get_paper_stats
from graph.graph_viz import GraphViz
from collect.paper_extension import PaperExpand


def router():
    pass

class PaperCollector:
    def __init(
            self,
            research_topics: Optional[List] = None,
            seed_paper_titles: Optional[Union[List[str], str]] = None,
            seed_paper_dois: Optional[Union[List[str], str]] = None,
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
            search_limit: Optional[int] = 50,
            recommend_limit: Optional[int] = 50,
            citation_limit: Optional[int] = 100,
            llm_api_key: Optional[str] = None,
            llm_model_name: Optional[str] = None,
            embed_api_key: Optional[str] = None,
            embed_model_name: Optional[str] = None,
            ):
        # seed papers info
        self.research_topics = research_topics
        self.seed_paper_titles = [seed_paper_titles] if isinstance(seed_paper_titles, str) and seed_paper_titles else seed_paper_titles if isinstance(seed_paper_titles, list) else []
        self.seed_paper_dois = [seed_paper_dois] if isinstance(seed_paper_dois, str) and seed_paper_dois else seed_paper_dois if isinstance(seed_paper_dois, list) else []

        # Filters
        self.from_dt = from_dt
        self.to_dt = to_dt
        self.fields_of_study = fields_of_study

        # LLM/Embedding
        self.llm_api_key = llm_api_key
        self.llm_model_name = llm_model_name
        self.embed_api_key = embed_api_key
        self.embed_model_name = embed_model_name

        # global status variables, which help track current progress and identify next step
        self.global_status = {
            'graph_stats':[],  # statistics for graph in each round
            'similar_papers':[],  # top similar papers in each round
            'key_authors':[],  # key authors in each round
            'crossref_papers':[],  # crossref papers in each round
        }
        
        # candidate nodes or edges
        self.candidates = {
            'candit_nodes_json': [],   # nodes json not included in the graph, but may potentially useful
            'candit_edges_json': [],   # edges json not included in the graph, but may potentially useful
        }

        # other parameters
        self.params = {
            # params for paper collection
            'search_limit': search_limit,
            'recommend_limit': recommend_limit,
            'citation_limit': citation_limit,
            # params for expanded search
            'similarity_threshold': 0.7,
            'top_k_similar_papers': 20,
            'top_l_key_authors': 20,
            # paprams for stopping criteria
            'min_paper_cnt': 50,
            'min_author_cnt': 50,
            'min_corssref_papers': 20,
            'min_key_authors': 10,
        }

        # initiate paper collector
        self.ps = PaperExpand(   
            research_topics = research_topics,   
            seed_paper_titles = seed_paper_titles, 
            seed_paper_dois = seed_paper_dois,
            llm_api_key = llm_api_key,
            llm_model_name = llm_model_name,
            embed_api_key = embed_api_key,
            embed_model_name = embed_model_name,
            from_dt = from_dt,
            to_dt = to_dt,
            fields_of_study = fields_of_study,
            search_limit = search_limit,
            recommend_limit = recommend_limit,
            citation_limit = citation_limit
            )

    def _smart_setup(self):
        # --- SETUP ---
        # for topics
        topics = []
        if candit_topics is not None and self.research_topics is not None:
            candit_topics = candit_topics + self.research_topics
        elif self.research_topics is not None:
            candit_topics = self.research_topics
        elif candit_topics is not None:
            candit_topics = candit_topics

        for topic in candit_topics:
            if topic not in self.ps.explored_nodes['topic']:
                topics.append(topic) 
        
        # for paper titles
        titles = []
        if candit_paper_dois is not None and self.seed_paper_titles is not None:
            candit_paper_titles = candit_paper_titles + self.seed_paper_titles
        elif self.seed_paper_titles is not None:
            candit_paper_titles = self.seed_paper_titles
        elif candit_paper_titles is not None:
            candit_paper_titles = candit_paper_titles

        for title in candit_paper_titles:
            if title not in self.ps.explored_nodes['title']:
                titles.append(title) 

        # for paper titles
        dois = []
        if candit_paper_dois is not None and self.candit_paper_dois is not None:
            candit_paper_dois = candit_paper_dois + self.seed_paper_titles
        elif self.seed_paper_titles is not None:
            candit_paper_dois = self.seed_paper_titles
        elif candit_paper_dois is not None:
            candit_paper_dois = candit_paper_dois

        for doi in candit_paper_dois:
            if doi not in self.ps.explored_nodes['paper']:
                dois.append(doi) 

        # set up parameters
        # search limit
        if len(dois) < 10 and len(titles) < 5 and len(topics) < 5:
            search_limit = 100
        else:
            search_limit = 50

        # time range
        # check current node status, check dt status
        # expand time range when needed

    async def _pruning_and_routing(
            self, 
            round,
            nodes_json, 
            edges_json,
            core_paper_dois,
            exclusion_paper_dois,
            exclusion_author_ids,
            ):
        stop_flag = 1
        unexplored_core_flag = 0
        insufficient_nodes_flag = 0
        insufficient_crossrefs_flag = 0
        unexplored_related_topics_flag = 0

        # --- Graph Stat ---
        G_pre = PaperGraph(name='Paper Graph Init Search')
        G_pre.add_graph_nodes(nodes_json)
        G_pre.add_graph_edges(edges_json)
        g_stat = get_graph_stats(G_pre)   # graph stats

        # valid paper with abstracts
        complete_paper_json = [node for node in nodes_json 
                               if node['labels'] == ['Paper'] 
                               and node['properties'].get('title') is not None and node['properties'].get('abstract') is not None]
        complete_paper_dois = [node['id'] for node in complete_paper_json]

        # core papers and core authors nodes
        core_author_ids = []
        for item in nodes_json:
            if item['id'] in core_paper_dois and isinstance(item['properties'].get('authors'), list):
                authors_id = [x['authorId'] for x in item['properties']['authors'] if x['authorId'] is not None] 
                core_author_ids.extend(authors_id)
        core_author_ids = list(set(core_author_ids))

        # --- SIMILARITY CALCULATION ---
        # check if similarity with edge type
        edge_types = [x[0] for x in g_stat['edge_type']]
        if 'SIMILAR_TO' not in edge_types:
            # calculate paper nodes similarity
            semantic_similar_pool = await self.ps.cal_embed_and_similarity(
                paper_nodes_json = complete_paper_json,
                paper_dois_1 = complete_paper_dois, 
                paper_dois_2 = complete_paper_dois,
                similarity_threshold=self.params['similarity_threshold'],
                )

            # prepare similarity edges
            sim_edgs_json = []
            for item in semantic_similar_pool:
                source_id = item['startNodeId']
                target_id = item['endNodeId']
                properties = item['properties']
                properties['relationshipType'] = item['relationshipType']
                sim_edgs_json.append((source_id, target_id, properties))  
            # add similarity edges to graph
            G_pre.add_graph_edges(sim_edgs_json)  

        # --- PRUNNING ---
        # pruning by connectivity
        sub_graphs = G_pre.find_wcc_subgraphs(target_nodes=core_paper_dois)
        if sub_graphs is not None and len(sub_graphs) > 0:
            G_post  = sub_graphs[0]
            # get stats after prunning
            g_stat = get_graph_stats(G_post)
        else:
            G_post = G_pre

        # --- GET KEY STATS ---
        # check paper count and author count
        paper_cnt, author_cnt = 0, 0
        for item in g_stat['node_type']:
            if item[0] == 'Paper':
                paper_cnt = item[1]
            elif item[0] == 'Author':
                author_cnt = item[1]

        # check core paper completeness
        # paper complete
        missing_doi = [doi for doi in core_paper_dois if doi not in G_post.nodes()]
        # citation complete
        missing_ref = [doi for doi in core_paper_dois if doi not in self.ps.explored_nodes['reference']]
        missing_citing = [doi for doi in core_paper_dois if doi not in self.ps.explored_nodes['citing']]
        # author complete
        missing_author = [aid for aid in core_author_ids if aid not in self.ps.explored_nodes['author']]

        paper_stats = get_paper_stats(G_post, exclusion_paper_dois)  # paper stats on graph
        author_stats = get_author_stats(G_post, exclusion_author_ids)  # author stats on graph

        # check crossref
        crossref_stats = []
        for x in paper_stats:
            if (x['if_exclude'] == False  # exclude seed papers 
                and x['local_citation_cnt'] > min(len(core_paper_dois),  5)):  # select most refered papers in graph
                crossref_stats.append(x)

        # check key authors
        key_authors_stats = []
        for x in author_stats:
            if (x['if_exclude'] == False  # exclude seed authors 
                and x['local_paper_cnt'] > min(len(core_paper_dois), 5)):  # select most refered papers in graph
                key_authors_stats.append(x)
        
        # check paper similarity
        sorted_paper_similarity = sorted(paper_stats, key=lambda x:x['max_sim_to_seed'], reverse=True)

        # related topics information
        # to be added

        # constraints on total # of nodes
        if paper_cnt < self.params['min_paper_cnt'] or author_cnt > self.params['min_author_cnt']:
            stop_flag = 0
            insufficient_nodes = 1
        # constraints on crossref papers
        if len(crossref_stats) < self.params['min_corssref_papers']:
            stop_flag = 0
            insufficient_crossrefs = 1

        # --- PROPOSE NEXT STEP ---
        if round == 1:
            # papers sufficient but 
            if insufficient_nodes == 1:
                
            else:
                if insufficient_crossrefs == 1:
        else:  # round > 1
            if insufficient_nodes == 1:
                self.to_dt = self.from_dt
                self.from_dt = datetime.strptime(self.from_dt, "%Y-%m-%d") - timedelta(weeks=52*5)
                
            and insufficient_nodes == 1:  
            


    async def seed_search(
            self, 
            round,
            candit_topics: Optional[List] = None,
            candit_paper_titles: Optional[Union[List[str], str]] = None,
            candit_paper_dois: Optional[Union[List[str], str]] = None,
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
            ):
        """Initial search for seed papers metadata"""
        # --- SETUP ---
        # for topics
        topics = []
        if candit_topics is not None and self.research_topics is not None:
            candit_topics = candit_topics + self.research_topics
        elif self.research_topics is not None:
            candit_topics = self.research_topics
        elif candit_topics is not None:
            candit_topics = candit_topics

        for topic in candit_topics:
            if topic not in self.ps.explored_nodes['topic']:
                topics.append(topic) 
        
        # for paper titles
        titles = []
        if candit_paper_dois is not None and self.seed_paper_titles is not None:
            candit_paper_titles = candit_paper_titles + self.seed_paper_titles
        elif self.seed_paper_titles is not None:
            candit_paper_titles = self.seed_paper_titles
        elif candit_paper_titles is not None:
            candit_paper_titles = candit_paper_titles

        for title in candit_paper_titles:
            if title not in self.ps.explored_nodes['title']:
                titles.append(title) 

        # for paper titles
        dois = []
        if candit_paper_dois is not None and self.candit_paper_dois is not None:
            candit_paper_dois = candit_paper_dois + self.seed_paper_titles
        elif self.seed_paper_titles is not None:
            candit_paper_dois = self.seed_paper_titles
        elif candit_paper_dois is not None:
            candit_paper_dois = candit_paper_dois

        for doi in candit_paper_dois:
            if doi not in self.ps.explored_nodes['paper']:
                dois.append(doi) 

        # set up parameters
        # search limit
        if len(dois) < 10 and len(titles) < 5 and len(topics) < 5:
            search_limit = 100
        else:
            search_limit = 50

        # time range
        # check current node status, check dt status
        # expand time range when needed

        # --- INITIAL QUERY on SEED ---
        # initial query for seed papers basic information
        print("--- Running Initial Query for Seed Papers Information ---")
        await self.ps.init_search(
            research_topics=topics,
            seed_paper_titles=titles,
            seed_paper_dois=dois,
            round=round,
            search_limit=search_limit,
            from_dt=from_dt,
            to_dt=to_dt,
            fields_of_study=fields_of_study
        )

        # get all paper infos
        paper_nodes_json = [node for node in self.ps.nodes_json 
                            if node['labels'] == ['Paper'] and 
                            node['properties'].get('title') is not None and node['properties'].get('abstract') is not None]
        paper_dois = [node['id'] for node in paper_nodes_json]

        # --- SIMILARITY CALCULATION ---
        # calculate paper nodes similarity
        semantic_similar_pool = await self.ps.cal_embed_and_similarity(
            paper_nodes_json=paper_nodes_json,
            paper_dois_1=paper_dois, 
            paper_dois_2=paper_dois,
            similarity_threshold=self.params['similarity_threshold'],
            )

        edges_json = semantic_similar_pool
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
            item['dataGeneration'] = {'round': 1, 'source': 'init_search'}


        # --- GRAPH PRUNING ---
        G_init = PaperGraph(name='Paper Graph Init Search')
        G_init.add_graph_nodes(ps.nodes_json)
        G_init.add_graph_edges(ps.edges_json)

        # stats
        graph_stats['init_search']['after_pruning'] = g_stat
        g_stat = get_graph_stats(G_init)

        # pruning
        sub_graphs = G_init.find_wcc_subgraphs(target_nodes=seed_dois)
        if sub_graphs is not None and len(sub_graphs) > 0:
            G_prune  = sub_graphs[0]
            g_stat = get_graph_stats(sub_graphs[0])
            graph_stats['init_search']['after_pruning'] = g_stat

        # --- FILTERING KEY NODES ---
        # paper stats
        paper_stats = get_paper_stats(G_init, seed_paper_dois)
        sorted_paper_similarity = sorted(paper_stats, key=lambda x:x['max_sim_to_seed'], reverse=True)
        candit_paper_dois = []
        filtered_papers_stats = []
        i = 0
        for x in sorted_paper_similarity:
            if i < 20:
                if (x['if_seed'] == False  # exclude seed papers 
                    and x['local_similarity_cnt'] > (len(paper_dois) / 5)):  # select most similar to others
                    candit_paper_dois.append(x['doi'])
                    filtered_papers_stats.append(x)
                    i += 1
            else:
                break
        similar_papers['init_search'] = filtered_papers_stats

        # author stats
        author_stats = get_author_stats(G_init, seed_author_ids)
        sorted_author_writes = sorted(author_stats, key=lambda x:x['local_paper_cnt'], reverse=True)
        filtered_authors = [x for x in sorted_author_writes if x['is_seed'] == False][0:top_l_key_authors]
        for item in filtered_authors:
            print(item)
        

    async def basic_search(self):
        # set up parameters
        citation_limit = 100
        if len(seed_dois) > 20 or len(seed_titles) > 20 or len(paper_dois) > 100:
            search_limit = 50
            recommend_limit = 50
        else :
            search_limit = 100
            recommend_limit = 100

        # --- DATA GENERATION ---
        print("--- Getting More Information Related to Seed Papers ---")
        # basic search for seed papers
        # may include seed paper authors, seed paper citation chain, recommendations based on seed papers 
        await ps.paper_search(
            seed_paper_dois=seed_paper_dois,
            seed_author_ids=seed_author_ids,
            search_citation = search_citation,
            search_author = search_author,
            round = 1,
            find_recommend = find_recommend,
            recommend_limit = recommend_limit,
            citation_limit = citation_limit,
            from_dt=ps.from_dt,
            to_dt=ps.to_dt,
            fields_of_study = ps.fields_of_study,
            )
        
        # get all paper infos
        paper_nodes_json = [node for node in ps.nodes_json 
                            if node['labels'] == ['Paper'] and 
                            node['properties'].get('title') is not None and node['properties'].get('abstract') is not None]
        paper_dois = [node['id'] for node in paper_nodes_json]

        # --- SIMILARITY CALCULATION ---
        # calculate paper nodes similarity
        semantic_similar_pool = await ps.cal_embed_and_similarity(
            paper_nodes_json=paper_nodes_json,
            paper_dois_1=paper_dois, 
            paper_dois_2=paper_dois,
            similarity_threshold=similarity_threshold,
            )

        edges_json = semantic_similar_pool
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
            item['dataGeneration'] = {'round': 1, 'source': 'init_search'}

       # --- GRAPH PRUNING ---
        G_init = PaperGraph(name='Paper Graph Init Search')
        G_init.add_graph_nodes(ps.nodes_json)
        G_init.add_graph_edges(ps.edges_json)

        # stats
        graph_stats['init_search']['after_pruning'] = g_stat
        g_stat = get_graph_stats(G_init)

        # pruning
        sub_graphs = G_init.find_wcc_subgraphs(target_nodes=seed_dois)
        if sub_graphs is not None and len(sub_graphs) > 0:
            G_prune  = sub_graphs[0]
            g_stat = get_graph_stats(sub_graphs[0])
            graph_stats['init_search']['after_pruning'] = g_stat

        # --- FILTERING KEY NODES ---
        # paper stats
        paper_stats = get_paper_stats(G_init, seed_paper_dois)
        sorted_paper_similarity = sorted(paper_stats, key=lambda x:x['max_sim_to_seed'], reverse=True)
        candit_paper_dois = []
        filtered_papers_stats = []
        i = 0
        for x in sorted_paper_similarity:
            if i < 20:
                if (x['if_seed'] == False  # exclude seed papers 
                    and x['local_similarity_cnt'] > (len(paper_dois) / 5)):  # select most similar to others
                    candit_paper_dois.append(x['doi'])
                    filtered_papers_stats.append(x)
                    i += 1
            else:
                break
        similar_papers['init_search'] = filtered_papers_stats

        # author stats
        author_stats = get_author_stats(G_init, seed_author_ids)
        sorted_author_writes = sorted(author_stats, key=lambda x:x['local_paper_cnt'], reverse=True)
        filtered_authors = [x for x in sorted_author_writes if x['is_seed'] == False][0:top_l_key_authors]
        for item in filtered_authors:
            print(item)

    async def expanded_search(self):
        # set up parameters
        crossref_info = crossref_papers['basic_search']

        candit_crossref_cnt = 0 
        for item in crossref_info:
            if item['local_citation_cnt'] > len(seed_dois):
                candit_crossref_cnt += 1
            else:
                break
        print(candit_crossref_cnt)

        # --- EXPAND to CITATIONS over SIMILAR PAPERS ---
        # get most similar papers to seed papers
        # track citation chain of these papers
        if if_expanded_citations is not None:
            print(f"\n--- Get crossref papers: ---")
            await ps.citation_expansion(
                seed_paper_dois = seed_paper_dois,
                candit_paper_dois = candit_paper_dois,  # user input of candit paper dois to search for citations
                search_citation = 'reference',
                round = 1,
                citation_limit = citation_limit,
                from_dt = ps.from_dt,
                to_dt = ps.to_dt,
                fields_of_study = ps.fields_of_study,
            )
        
        # get all paper infos
        paper_nodes_json = [node for node in ps.nodes_json 
                            if node['labels'] == ['Paper'] and 
                            node['properties'].get('title') is not None and node['properties'].get('abstract') is not None]
        paper_dois = [node['id'] for node in paper_nodes_json]

        # --- SIMILARITY CALCULATION ---
        # calculate paper nodes similarity
        semantic_similar_pool = await ps.cal_embed_and_similarity(
            paper_nodes_json=paper_nodes_json,
            paper_dois_1=paper_dois, 
            paper_dois_2=paper_dois,
            similarity_threshold=similarity_threshold,
            )

        edges_json = semantic_similar_pool
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
            item['dataGeneration'] = {'round': 1, 'source': 'init_search'}

       # --- GRAPH PRUNING ---
        G_init = PaperGraph(name='Paper Graph Init Search')
        G_init.add_graph_nodes(ps.nodes_json)
        G_init.add_graph_edges(ps.edges_json)

        # stats
        graph_stats['init_search']['after_pruning'] = g_stat
        g_stat = get_graph_stats(G_init)

        # pruning
        sub_graphs = G_init.find_wcc_subgraphs(target_nodes=seed_dois)
        if sub_graphs is not None and len(sub_graphs) > 0:
            G_prune  = sub_graphs[0]
            g_stat = get_graph_stats(sub_graphs[0])
            graph_stats['init_search']['after_pruning'] = g_stat

        # --- FILTERING KEY NODES ---
        # paper stats
        paper_stats = get_paper_stats(G_init, seed_paper_dois)
        sorted_paper_similarity = sorted(paper_stats, key=lambda x:x['max_sim_to_seed'], reverse=True)
        candit_paper_dois = []
        filtered_papers_stats = []
        i = 0
        for x in sorted_paper_similarity:
            if i < 20:
                if (x['if_seed'] == False  # exclude seed papers 
                    and x['local_similarity_cnt'] > (len(paper_dois) / 5)):  # select most similar to others
                    candit_paper_dois.append(x['doi'])
                    filtered_papers_stats.append(x)
                    i += 1
            else:
                break
        similar_papers['init_search'] = filtered_papers_stats

        # author stats
        author_stats = get_author_stats(G_init, seed_author_ids)
        sorted_author_writes = sorted(author_stats, key=lambda x:x['local_paper_cnt'], reverse=True)
        filtered_authors = [x for x in sorted_author_writes if x['is_seed'] == False][0:top_l_key_authors]
        for item in filtered_authors:
            print(item)



async def test():
    # driving examples
    llm_api_key = os.getenv('GEMINI_API_KEY_3')
    llm_model_name="gemini-2.0-flash"
    embed_api_key = os.getenv('GEMINI_API_KEY_3')
    embed_model_name="models/text-embedding-004"

    research_topics = ["llm literature review"]
    seed_dois = ['10.48550/arXiv.2406.10252',  # AutoSurvey: Large Language Models Can Automatically Write Surveys
                '10.48550/arXiv.2412.10415',  # Generative Adversarial Reviews: When LLMs Become the Critic
                '10.48550/arXiv.2402.12928',  # A Literature Review of Literature Reviews in Pattern Analysis and Machine Intelligence 
                ]
    seed_titles = ['PaperRobot: Incremental Draft Generation of Scientific Ideas',
                'From Hypothesis to Publication: A Comprehensive Survey of AI-Driven Research Support Systems'
                ]
        