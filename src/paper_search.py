import asyncio
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Literal # Added Tuple

import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from models.embedding_models import gemini_embedding_async, semantic_similarity_matrix

from collect.paper_query import PaperQuery
from collect.citation_query import CitationQuery
from collect.author_query import AuthorQuery
from collect.paper_recommendation import PaperRecommendation

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PaperFinder:
    """
    A class for exploring academic papers using Semantic Scholar API and LLMs,
    optimized for asynchronous operations.
    """
    def __init__(
            self,
            research_topic: Optional[str] = None,
            seed_paper_titles: Optional[Union[List[str], str]] = None,
            seed_paper_dois: Optional[Union[List[str], str]] = None,
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
            nodes_json: Optional[List[Dict]] = None,
            edges_json: Optional[List[Dict]] = None, 
            search_limit: Optional[int] = 50,
            recommend_limit: Optional[int] = 50,
            citation_limit: Optional[int] = 100
    ):
        """
        Initialize PaperExploration parameters.
        (Args documentation omitted for brevity, same as original)
        """
        # seed papers info
        self.research_topic = research_topic
        self.seed_paper_titles = [seed_paper_titles] if isinstance(seed_paper_titles, str) and seed_paper_titles else seed_paper_titles if isinstance(seed_paper_titles, list) else []
        self.seed_paper_dois = [seed_paper_dois] if isinstance(seed_paper_dois, str) and seed_paper_dois else seed_paper_dois if isinstance(seed_paper_dois, list) else []

        self.search_limit = search_limit
        self.recommend_limit = recommend_limit
        self.citation_limit = citation_limit

        # Filters
        self.from_dt = from_dt
        self.to_dt = to_dt
        self.fields_of_study = fields_of_study

        # State: Nodes and Edges
        self.nodes_json = []
        if isinstance(nodes_json, list) and len(nodes_json) > 0:
            self._add_items_json(nodes_json)
            
        self.edges_json = []
        if isinstance(edges_json, list) and len(edges_json) > 0:
            self._add_items_json(edges_json)

        # store paper abstraction embeddings
        self.abs_embed_ref = {}

        # explored nodes
        self.seeds = {'paper':[], 'author':[]} # seed papers and authors
        self.explored_nodes = {'topic':[],   # topic explored
                               'author':[],   # authors explored 
                               'reference':[],  # papers with reference explored
                               'citing': []  # papers with citation explored
                                }


    ############################################################################
    # basic function
    ############################################################################
    def _add_items_json(
            self, 
            items: List[Dict], 
            round: Optional[int]=1,
            source: Optional[str]=None):
        """Adds nodes and relationships from processed data, remove duplications
        Args:
            items: node json or edge json
            round / source: indicating from which iteration the node or edge was created.
                            round and source help to track the dynamics of the data generation
        """
        _node_ids = set()
        for x in self.nodes_json:
            # for author and paper, avoid duplication if there is complete information
            if x['labels'][0] in ['Author', 'Paper']:
                if x.get('properties', {}).get('is_complete', False) == True:
                    _node_ids.add(x['id'])
            # for others, just avoid duplication
            else:
                _node_ids.add(x['id'])
        _edge_tuples = set([(x['startNodeId'], x['endNodeId'], x['relationshipType']) 
                            for x in self.edges_json]) # Store (start_id, end_id, type) tuples
        
        for item in items:
            if item['type'] == 'node':
                node_id = item['id']
                if node_id not in _node_ids:
                    item['dataGeneration'] = {'round': round, 'source': source}
                    self.nodes_json.append(item)
                    if item['labels'][0] in ['Author', 'Paper']:
                        if item.get('properties', {}).get('is_complete', False) == True:
                            _node_ids.add(item['id'])
                    else:
                        _node_ids.add(item['id'])

            elif item['type'] == 'relationship':
                rel_type = item['relationshipType']
                start_id = item['startNodeId']
                end_id = item['endNodeId']
                edge_tuple = (start_id, end_id, rel_type) 
                if edge_tuple not in _edge_tuples:
                    item['dataGeneration'] = {'round': round, 'source': source}
                    self.edges_json.append(item)
                    _edge_tuples.add(edge_tuple)

 
    ############################################################################
    # paper collection functions
    ############################################################################
    async def init_search(
            self,
            research_topic: Optional[str]=None,
            seed_paper_titles: Optional[List]=None,
            seed_paper_dois: Optional[List]=None,
            round: Optional[int] = 1,
            search_limit: Optional[int] = 50,
            from_dt: Optional[str]="2000-01-01",
            to_dt: Optional[str]="9999-12-31"
        ):
        """basic search for seed paper related information"""
        research_topic = research_topic if research_topic else self.research_topic
        seed_paper_titles = seed_paper_titles if seed_paper_titles else self.seed_paper_titles
        seed_paper_dois = seed_paper_dois if seed_paper_dois else self.seed_paper_dois

        pq = PaperQuery()
        initial_papers_json = await pq.get_paper_info(
            research_topic=research_topic, 
            seed_paper_titles=seed_paper_titles,
            seed_paper_dois=seed_paper_dois,
            limit=search_limit, from_dt=from_dt, to_dt=to_dt)
        
        # add item as json
        self._add_items_json(initial_papers_json, round, source='init_search')
        logging.info(f"Graph state after initial search tasks. Nodes: {len(self.nodes_json)}, Edges: {len(self.edges_json)}")

        # get seed DOIs
        seed_paper_dois = [node['id'] for node in self.nodes_json if node['labels'] == ['Paper'] and node['properties'].get('from_seed')==True]
        seed_author_ids = []
        for node in self.nodes_json:
            if node['labels'] == ['Paper'] and node['properties'].get('from_seed')==True and isinstance(node['properties'].get('authors'), list):
                authors_id = [x['authorId'] for x in node['properties']['authors'] if x['authorId'] is not None] 
                seed_author_ids.extend(authors_id)
        self.seeds['paper'].extend(seed_paper_dois)
        self.seeds['author'].extend(seed_author_ids)

        # update explored nodes
        if research_topic is not None:
            self.explored_nodes['topic'].extend([research_topic])


    async def paper_search(
            self,
            seed_paper_dois: Optional[List[str]] = None,
            seed_author_ids: Optional[List[str]] = None,
            search_citation: Literal['reference', 'citing', 'both'] = None,
            search_author: Optional[bool] = False,
            round: Optional[int] = 1,
            find_recommend: Optional[bool] = False,
            recommend_limit: Optional[int] = 50,
            citation_limit: Optional[int] = 100,
            from_dt='2022-01-01',
            to_dt='2025-03-13',
            fields_of_study=None,
    ):
        """
        Main asynchronous collection method orchestrating various data fetching tasks.
        """
        if (search_citation is not None or find_recommend == True) and not seed_paper_dois:
             logging.error("Search citation or paper recommendation with no seed paper DOIs.")
             return [], []

        if search_author == True and not seed_author_ids:
             logging.error("Search author called with no seed author IDs.")
             return [], []

        tasks = []
        processed_items_aggregate = [] # Store results from tasks

        # --- Task Prep: Author Info ---
        if search_author:
            if seed_author_ids and isinstance(seed_author_ids, list):
                 logging.info(f"Preparing author info task for {len(seed_author_ids)} authors.")
                 aq = AuthorQuery()
                 tasks.append(aq.get_author_info(seed_author_ids, from_dt, to_dt, fields_of_study))
            else:
                 logging.info("No new authors found for seed DOIs or all are already complete.")

        # --- Task Prep: References & Citations ---
        if search_citation and seed_paper_dois:
            logging.info(f"Preparing reference/citation tasks for {len(seed_paper_dois)} seed papers.")
            cq = CitationQuery()
            for paper_doi in seed_paper_dois:
                if search_citation in ['reference', 'both']:  # cited papers only
                    tasks.append(cq.get_cited_papers(paper_doi, citation_limit, from_dt, to_dt, fields_of_study))
                if search_citation in ['citing', 'both']:  # cited papers only
                    tasks.append(cq.get_citing_papers(paper_doi, citation_limit, from_dt, to_dt, fields_of_study))

        # --- Task Prep: Recommendations ---
        if find_recommend and seed_paper_dois:
             logging.info("Preparing recommendations task.")
             rq = PaperRecommendation()
             tasks.append(rq.get_recommend_papers(seed_paper_dois, recommend_limit, from_dt, to_dt, fields_of_study))

        # --- Execute Concurrent Data Fetching Tasks (excluding expanded search for now) ---
        if tasks:
            logging.info(f"Running {len(tasks)} main data collection tasks concurrently...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results immediately
            for result in results:
                if isinstance(result, Exception):
                    logging.error(f"Data collection task failed: {result}")
                elif isinstance(result, list):
                    processed_items_aggregate.extend(result) # Add items from successful tasks
                    logging.info("Main data collection tasks finished.")
        else:
            logging.info("No main data collection tasks to run.")

        # --- Add all aggregated items to the graph state ---
        self._add_items_json(processed_items_aggregate, round, source='basic_search')
        logging.info(f"Graph state after collection tasks. Nodes: {len(self.nodes_json)}, Edges: {len(self.edges_json)}")
        
        # update explored status
        if search_citation in ['reference', 'both']:
            self.explored_nodes['reference'].extend(seed_paper_dois) 
        if search_citation in ['citing', 'both']:
            self.explored_nodes['citing'].extend(seed_paper_dois) 
        if search_author:
            self.explored_nodes['author'].extend(seed_author_ids) 

