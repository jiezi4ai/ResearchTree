
import copy
from typing import List, Dict, Optional, Union, Tuple, Literal # Added Tuple

import os
import json

import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from graph.paper_graph import PaperGraph
from graph.graph_viz import GraphViz
from collect.paper_extension import PaperExpand


class PaperCollector:
    def __init(self):
        graph_evolution = {}

        graph_stats = {}

        similarity_threshold = 0.7
        top_k_similar_papers = 20
        similar_papers = {}

        top_l_key_authors = 20
        key_authors = {}

        crossref_papers = {}
        top_m_corssref_papers = 20

        candit_edges_pool = []

        # driving examples
        llm_api_key = os.getenv('GEMINI_API_KEY_3')
        llm_model_name="gemini-2.0-flash"
        embed_api_key = os.getenv('GEMINI_API_KEY_3')
        embed_model_name="models/text-embedding-004"

        research_topic = "llm literature review"
        seed_dois = ['10.48550/arXiv.2406.10252',  # AutoSurvey: Large Language Models Can Automatically Write Surveys
                    '10.48550/arXiv.2412.10415',  # Generative Adversarial Reviews: When LLMs Become the Critic
                    '10.48550/arXiv.2402.12928',  # A Literature Review of Literature Reviews in Pattern Analysis and Machine Intelligence 
                    ]
        seed_titles = ['PaperRobot: Incremental Draft Generation of Scientific Ideas',
                    'From Hypothesis to Publication: A Comprehensive Survey of AI-Driven Research Support Systems'
                    ]
        
        ps = PaperCollector(   
            research_topic = research_topic,   
            seed_paper_titles = seed_titles, 
            seed_paper_dois = seed_dois,
            llm_api_key = llm_api_key,
            llm_model_name = llm_model_name,
            embed_api_key = embed_api_key,
            embed_model_name = embed_model_name,
            from_dt = '2020-01-01',
            to_dt = '2025-04-30',
            fields_of_study = ['Computer Science'],
            search_limit = search_limit,
            recommend_limit = recommend_limit,
            citation_limit = citation_limit
            )
        
        pg = PaperGraph()
        

    async def init_search(self):
        # set up parameters

        citation_limit = 100

        if len(seed_dois) < 10 or len(seed_titles) < 10:
            search_limit = 100
            recommend_limit = 100
        else:
            search_limit = 50
            recommend_limit = 50

        # --- INITIAL QUERY on SEED ---
        # initial query for seed papers basic information
        print("--- Running Initial Query for Seed Papers Information ---")
        await ps.init_search(
            research_topic=ps.research_topic,
            seed_paper_titles=ps.seed_paper_titles,
            seed_paper_dois=ps.seed_paper_dois,
            round=round,
            search_limit=ps.search_limit,
            from_dt=ps.from_dt,
            to_dt=ps.to_dt
        )

        # get all paper infos
        paper_nodes_json = [node for node in ps.nodes_json 
                            if node['labels'] == ['Paper'] and 
                            node['properties'].get('title') is not None and node['properties'].get('abstract') is not None]
        paper_dois = [node['id'] for node in paper_nodes_json]

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

        # basic stats
        G_init = PaperGraph(name='Paper Graph Init Search')
        G_init.add_graph_nodes(ps.nodes_json)
        G_init.add_graph_edges(ps.edges_json)
        g_stat = get_graph_stats(G_init)