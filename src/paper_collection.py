import asyncio
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Literal # Added Tuple

import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from paper_expansion import PaperExpansion

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PaperCollection(PaperExpansion):
    """
    A class for exploring academic papers using Semantic Scholar API and LLMs,
    optimized for asynchronous operations.
    """
    def __init__(
            self,
            research_topic: Optional[str] = None,
            seed_paper_titles: Optional[Union[List[str], str]] = None,
            seed_paper_dois: Optional[Union[List[str], str]] = None,
            llm_api_key: Optional[str] = None,
            llm_model_name: Optional[str] = None,
            embed_api_key: Optional[str] = None,
            embed_model_name: Optional[str] = None,
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
            nodes_json: Optional[List[Dict]] = None,
            edges_json: Optional[List[Dict]] = None, 
            search_limit: Optional[int] = 50,
            recommend_limit: Optional[int] = 50,
            citation_limit: Optional[int] = 100,
            paper_graph_name: Optional[str] = 'paper_graph'
    ):
        """
        Initialize PaperExploration parameters.
        (Args documentation omitted for brevity, same as original)
        """
        super().__init__(
            research_topic=research_topic,
            seed_paper_titles=seed_paper_titles,
            seed_paper_dois=seed_paper_dois,
            from_dt=from_dt,
            to_dt=to_dt,
            fields_of_study=fields_of_study,
            nodes_json=nodes_json,
            edges_json=edges_json, 
            search_limit=search_limit,
            recommend_limit=recommend_limit,
            citation_limit=citation_limit,
            paper_graph_name=paper_graph_name,
            llm_api_key = llm_api_key,
            llm_model_name = llm_model_name,
            embed_api_key = embed_api_key,
            embed_model_name = embed_model_name,            
            )
     

    async def paper_collection(
            self,
            search_citation: Literal['reference', 'citing', 'both'] = None,
            search_author: Optional[bool] = False,
            find_recommend: Optional[bool] = False,
            if_related_topic: Optional[bool] = False,
            if_expanded_citations: Literal['reference', 'citing', 'both'] = None,
            candit_expand_paper_dois: Optional[List] = None,
            candit_expand_author_ids: Optional[List] = None,
            if_expanded_authors: Optional[bool] = False,
            if_include_seed_author: Optional[bool] = False,
            if_add_similarity: Optional[bool] = False,
            similarity_threshold: Optional[float] = 0.7,
            expanded_k_papers: Optional[int] = 10,
            expanded_l_authors: Optional[int] = 50,
            ):
        # --- INITIAL QUERY on SEED ---
        # initial query for seed papers basic information
        print("--- Running Initial Query for Seed Papers Information ---")
        await self.init_search(
            self.research_topic,
            self.seed_paper_titles,
            self.seed_paper_dois,
            self.search_limit,
            self.from_dt,
            self.to_dt
        )

        # get seed DOIs
        seed_paper_dois = [node['id'] for node in self.nodes_json if node['labels'] == ['Paper'] and node['properties'].get('from_seed')==True]
        seed_author_ids = []
        for node in self.nodes_json:
            if node['labels'] == ['Paper'] and node['properties'].get('from_seed')==True and isinstance(node['properties'].get('authors'), list):
                authors_id = [x['authorId'] for x in node['properties']['authors'] if x['authorId'] is not None] 
                seed_author_ids.extend(authors_id)
        seed_paper_json = [node for node in self.nodes_json if node['labels'] == ['Paper'] and node['properties'].get('from_seed')==True]
        self.explored_nodes['seed'].extend(seed_paper_dois) 

        # --- MORE INFORMATION on SEED ---
        print("--- Getting More Information Related to Seed Papers ---")
        # basic search for seed papers
        # may include seed paper authors, seed paper citation chain, recommendations based on seed papers 
        await self.paper_search(
            seed_paper_dois=seed_paper_dois,
            seed_author_ids=seed_author_ids,
            search_citation = search_citation,
            search_author = search_author,
            find_recommend = find_recommend,
            recommend_limit = self.recommend_limit,
            citation_limit = self.citation_limit,
            from_dt=self.from_dt,
            to_dt=self.to_dt,
            fields_of_study = self.fields_of_study,
            )
        if search_citation in ['reference', 'both']:
            self.explored_nodes['reference'].extend(seed_paper_dois) 
        if search_citation in ['citing', 'both']:
            self.explored_nodes['citing'].extend(seed_paper_dois) 
        if search_author:
            self.explored_nodes['author'].extend(seed_author_ids) 

        # --- EXPAND to RELATED TOPICS over SEED ---
        # get related topics based on abstracts of seed papers
        # search for related topics for more papers
        print("--- Extend Related Topics from Seed Papers ---")
        if if_related_topic:
            await self.topic_extension(
                seed_paper_json=seed_paper_json, 
                llm_api_key=self.llm_api_key, 
                llm_model_name=self.llm_model_name,
                search_limit=self.search_limit,
                from_dt=self.from_dt,
                to_dt=self.to_dt,
                fields_of_study = self.fields_of_study,
                )

        # --- INTERMEDIATE: CALCULATE SIMILARITY ---
        # get all paper infos
        paper_nodes_json = [node for node in self.nodes_json 
                            if node['labels'] == ['Paper'] and 
                            node['properties'].get('title') is not None and node['properties'].get('abstract') is not None]
        paper_dois = [node['id'] for node in paper_nodes_json]

        # calculate paper nodes similarity
        semantic_similar_pool = await self.cal_embed_and_similarity(
            paper_nodes_json=paper_nodes_json,
            paper_dois_1=paper_dois, 
            paper_dois_2=paper_dois,
            similarity_threshold=similarity_threshold,
        )

        # --- EXPAND to CITATIONS over SIMILAR PAPERS ---
        # get most similar papers to seed papers
        # track citation chain of these papers
        if if_expanded_citations is not None:
            print(f"\n--- Get crossref papers: ---")
            await self.citation_expansion(
                seed_paper_dois = seed_paper_dois,
                semantic_similar_pool = semantic_similar_pool,  # from self.cal_embed_and_similarity
                candit_paper_dois = candit_expand_paper_dois,  # user input of candit paper dois to search for citations
                expanded_k_papers = expanded_k_papers,
                similarity_threshold = similarity_threshold,
                search_citation = 'reference',
                citation_limit = self.citation_limit,
                from_dt = self.from_dt,
                to_dt = self.to_dt,
                fields_of_study = self.fields_of_study,
            )

        # --- EXPAND SIGNIFICANT AUTHORS ---
        # filter most refered papers from graph
        # then search for author information
        if if_expanded_authors:
            print(f"\n--- Get most cited authors: ---")
            await self.author_expansion(
                seed_author_ids = seed_author_ids,
                candit_author_ids = candit_expand_author_ids,
                expanded_l_authors = expanded_l_authors,
                if_include_seed_author = if_include_seed_author,
                from_dt = self.from_dt,
                to_dt = self.to_dt,
                fields_of_study = self.fields_of_study,
            )

        # --- CALCULATE PAPERS SIMILARITY ---
        if if_add_similarity:
            # get all paper nodes
            paper_nodes_json = [node for node in self.nodes_json 
                                if node['labels'] == ['Paper'] and 
                                node['properties'].get('title') is not None and node['properties'].get('abstract') is not None]
            paper_ids = [node['id'] for node in paper_nodes_json]

            # calculate paper nodes similarity
            semantic_similar_pool = await self.cal_embed_and_similarity(
                paper_nodes_json=paper_nodes_json,
                paper_dois_1=paper_ids, 
                paper_dois_2=paper_ids,
                similarity_threshold=similarity_threshold,
            )

            self._add_items_to_graph(semantic_similar_pool)
            logging.info(f"Graph state after expansion tasks. Nodes: {len(self.nodes_json)}, Edges: {len(self.edges_json)}")