import asyncio
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Literal # Added Tuple

import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from models.embedding_models import gemini_embedding_async, semantic_similarity_matrix

from collect.citation_query import CitationQuery
from collect.author_query import AuthorQuery
from collect.related_topic_query import RelatedTopicQuery

from paper_search import PaperSearch

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PaperExpand(PaperSearch):
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
            citation_limit: Optional[int] = 100
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
            citation_limit=citation_limit
            )

        # LLM/Embedding
        self.llm_api_key = llm_api_key
        self.llm_model_name = llm_model_name
        self.embed_api_key = embed_api_key
        self.embed_model_name = embed_model_name
        

    ############################################################################
    # paper similarity calculation
    ############################################################################
    async def get_abstract_embeds(
            self, 
            paper_nodes_json, 
            embed_api_key: Optional[str] = None,
            embed_model_name: Optional[str] = None,
            ):
        """Calculate semantic similarity asynchronously."""
        ids, texts = [], []
        for node in paper_nodes_json:
            node_id = node['id']
            title = node['properties'].get('title')
            abstract = node['properties'].get('abstract')
            if title is not None and abstract is not None:
                texts.append(f"TITLE: {title} \nABSTRACT: {abstract}")
                ids.append(node_id)

        if not texts:
            logging.warning("No paper titles and abstracts found for semantic similarity calculation.")
            return {}

        logging.info(f"Generating embeddings for {len(texts)} papers...")
        # Assume gemini_embedding_async handles its own batching/rate limits if necessary
        try:
             embeds = await gemini_embedding_async(embed_api_key, embed_model_name, texts, 10) # Pass batch size
        except Exception as e:
             logging.error(f"Embedding generation failed: {e}")
             return {}

        if embeds is None or (len(embeds) != len(texts)):
             logging.error(f"Embedding generation returned unexpected result. Expected {len(texts)} embeddings, got {len(embeds) if embeds else 0}.")
             return {}
        
        abs_embed_ref = {key: value for key, value in zip(ids, embeds)}    
        return abs_embed_ref
    
    def cal_paper_similarity(
            self,
            paper_nodes_json,
            paper_dois_1, 
            paper_dois_2,
            abs_embed_ref,
            similarity_threshold,
            ):
        semantic_similar_pool = []
        publish_dt_ref = {x['id']:x['properties'].get('publicationDate')
                        for x in paper_nodes_json if x['properties'].get('publicationDate') is not None}

        # Filter embeddings for each list
        embeds_ref_1 = {key: val for key, val in abs_embed_ref.items() if key in paper_dois_1}
        embeds_ref_2 = {key: val for key, val in abs_embed_ref.items() if key in paper_dois_2}

        # --- Start: Add Robustness Checks ---
        ids_1 = list(embeds_ref_1.keys())
        ids_2 = list(embeds_ref_2.keys())
        embed_values_1 = list(embeds_ref_1.values())
        embed_values_2 = list(embeds_ref_2.values())

        if not embed_values_1 or not embed_values_2:
            logging.warning(f"Cannot calculate similarity. "
                            f"Found {len(embed_values_1)} valid embeddings for list 1 (IDs: {ids_1}) and "
                            f"{len(embed_values_2)} for list 2 (IDs: {ids_2}). "
                            f"Check if input DOIs exist and have title/abstract.")
            return [] # Return empty list as no similarity can be calculated
        # --- End: Add Robustness Checks ---

        embeds_1 = np.array(embed_values_1)
        embeds_2 = np.array(embed_values_2)

        # Add logging for shapes *before* the calculation
        logging.info(f"Shape of embeds_1: {embeds_1.shape}")
        logging.info(f"Shape of embeds_2: {embeds_2.shape}")

        logging.info("Calculating similarity matrix...")
        try:
            # Assuming semantic_similarity_matrix handles potential normalization etc.
            # It likely computes cosine similarity: (embeds_1 @ embeds_2.T) / (norm(embeds_1) * norm(embeds_2))
            sim_matrix = semantic_similarity_matrix(embeds_1, embeds_2)
            sim_matrix = np.array(sim_matrix) # Ensure it's a numpy array if not already
        except Exception as e:
            # Log the shapes again in case of error
            logging.error(f"Similarity matrix calculation failed with embeds_1 shape {embeds_1.shape} and embeds_2 shape {embeds_2.shape}: {e}")
            return [] # Return empty list on failure

        logging.info("Processing similarity matrix to create relationships...")
        rows, cols = sim_matrix.shape
        added_pairs = set()

        # --- Small optimization/correction in loop ---
        if rows > 0 and cols > 0:
            # Ensure sim_matrix shape matches expectation: (len(ids_1), len(ids_2))
            if sim_matrix.shape != (len(ids_1), len(ids_2)):
                logging.error(f"Similarity matrix shape {sim_matrix.shape} does not match expected shape ({len(ids_1)}, {len(ids_2)})")
                return []

            for i in range(rows):      # Iterate through papers in list 1
                id_i = ids_1[i]
                publish_dt_i = publish_dt_ref.get(id_i)
                if publish_dt_i is None: # Skip if no publication date for comparison
                    continue

                for j in range(cols):  # Iterate through papers in list 2
                    id_j = ids_2[j]
                    # Avoid self-comparison if the lists could overlap and contain the same ID
                    if id_i == id_j:
                        continue

                    sim = sim_matrix[i, j]
                    if sim > similarity_threshold:
                        publish_dt_j = publish_dt_ref.get(id_j)
                        if publish_dt_j is None: # Skip if no publication date for comparison
                            continue

                        # Determine start/end based on publication date
                        if publish_dt_i <= publish_dt_j:
                            start_node_id = id_i
                            end_node_id = id_j
                        else:
                            start_node_id = id_j
                            end_node_id = id_i

                        # Create unique tuple for the pair (order matters for the relationship direction)
                        pair_tuple = (start_node_id, end_node_id)

                        if pair_tuple not in added_pairs:
                            edge = {
                                "type": "relationship",
                                "relationshipType": "SIMILAR_TO",
                                "startNodeId": start_node_id,
                                "endNodeId": end_node_id,
                                "properties": {
                                    'source': 'semantic similarity',
                                    'weight': round(float(sim), 4),
                                }
                            }
                            semantic_similar_pool.append(edge)
                            added_pairs.add(pair_tuple) # Store the directed pair
        else:
            logging.info("Similarity matrix is empty, no relationships to process.")

        return semantic_similar_pool
    
    async def cal_embed_and_similarity(
            self, 
            paper_nodes_json,
            paper_dois_1, 
            paper_dois_2,
            similarity_threshold,
            ):
        """calculate embeds and similarity"""
        paper_ids = [node['id'] for node in paper_nodes_json]
        input_paper_dois = list(set(paper_dois_1).union(paper_dois_2))

        status_1 = bool(set(paper_dois_1).intersection(set(paper_ids)))
        status_2 = bool(set(paper_dois_2).intersection(set(paper_ids)))
        if status_1 and status_2:
            # filter those not yet have emebedings
            paper_nodes_json_wo_embeds = [node for node in paper_nodes_json 
                                          if node['id'] in input_paper_dois
                                          and node['id'] not in self.abs_embed_ref]
            
            # calculate embeddings for those nodes
            if len(paper_nodes_json_wo_embeds) > 0:
                abs_embed_ref = await self.get_abstract_embeds(paper_nodes_json_wo_embeds, self.embed_api_key, self.embed_model_name) 
                # add embeddings
                self.abs_embed_ref.update(abs_embed_ref)

            semantic_similar_pool = self.cal_paper_similarity(
                paper_nodes_json, 
                paper_dois_1=paper_dois_1,
                paper_dois_2=paper_dois_2,
                abs_embed_ref=self.abs_embed_ref,
                similarity_threshold=similarity_threshold,
            )
            return semantic_similar_pool
        else:
            logging.error(f"The input {'' if status_1 else '1'} {'' if status_2 else '2'} paper dois have no intersection with current paper json.")
            return []


    ############################################################################
    # paper collection functions
    ############################################################################
    async def topic_generation(
            self,
            seed_paper_json: Optional[List[Dict]],
            llm_api_key: Optional[str] = None,
            llm_model_name: Optional[str] = None,
            round: Optional[int] = 1,
            ):
        """use LLM to generate related topics based on seed paper information"""
        logging.info("Use LLM to identify key related topics.")
        tq = RelatedTopicQuery(llm_api_key=llm_api_key, llm_model_name=llm_model_name)
        keywords_topics_json = await tq.llm_gen_related_topics(seed_paper_json)
        topics_json = tq.get_topic_json(keywords_topics_json, seed_paper_json)

        # add topic to json
        self._add_items_json(topics_json, round, source="topic_extention")

        return keywords_topics_json


    async def topic_extension(
            self,
            candit_topic_queries: Optional[List[str]] = None,
            round: Optional[int] = 1,
            search_limit: Optional[int] = 50,
            from_dt='2022-01-01',
            to_dt='2025-03-13',
            fields_of_study=None,
    ):
        if not isinstance(candit_topic_queries, list) or len(candit_topic_queries) == 0:
            logging.warning("No candit topics from input.")
            topic_results = []

        # exclude topic already explored
        topic_queries = []
        for query in candit_topic_queries:
            if query not in self.explored_nodes['topic']:
                topic_queries.append(query)

        if isinstance(topic_queries, list) and len(topic_queries) > 0:
            logging.info(f"Expand to {len(topic_queries)} related topics for more papers.")
            tq = RelatedTopicQuery()
            topic_results = await tq.get_related_papers(topic_queries, search_limit, from_dt, to_dt, fields_of_study)
        else:
            logging.info("No new topics found or all are already complete.")
            topic_results = []

        # --- Add all aggregated items to the graph state ---
        self._add_items_json(topic_results, round, source="topic_extention")
        logging.info(f"Graph state after expansion tasks. Nodes: {len(self.nodes_json)}, Edges: {len(self.edges_json)}")

        # add to explored topics
        if len(topic_queries) > 0:
            self.explored_nodes['topic'].extend([topic_queries])

    async def citation_expansion(
            self,
            seed_paper_dois,
            candit_paper_dois: Optional[List] = None,  # user input of candit paper dois to search for citations
            search_citation: Literal['reference', 'citing', 'both'] = 'reference',
            if_include_seed_paper: Optional[bool] = False,
            round: Optional[int] = 1,
            citation_limit: Optional[int] = 100,
            from_dt='2022-01-01',
            to_dt='2025-03-13',
            fields_of_study=None,
            ):
        """expand citation chain based on most similar papers
        - filter k papers based on semantic similarity to seed papers
        - search citations for these papers
        The citation for similar papers would help consolidate cross reference.
        """
        if not isinstance(candit_paper_dois, list) or len(candit_paper_dois) == 0:
            logging.warning("No candit papers from input.")
            citation_results = []

        # whether exclude seed paper
        if not if_include_seed_paper:
            candit_paper_dois = [x for x in candit_paper_dois if x not in seed_paper_dois]

        # exclude papers that already with complete information
        ref_paper_dois = []
        if search_citation in ['reference', 'both']:
            for doi in candit_paper_dois:
                if doi not in self.explored_nodes['reference']:
                    ref_paper_dois.append(doi)

        citing_paper_dois = []
        if search_citation in ['citing', 'both']:
            for doi in candit_paper_dois:
                if doi not in self.explored_nodes['citing']:
                    citing_paper_dois.append(doi)

        tasks = []
        citation_results = [] # Store results from tasks

        # --- Task Prep: References & Citations ---
        logging.info(f"Preparing reference/citation tasks for extended papers.")
        cq = CitationQuery()
        if len(ref_paper_dois) > 0: 
            for paper_doi in ref_paper_dois:
                tasks.append(cq.get_cited_papers(paper_doi, citation_limit, from_dt, to_dt, fields_of_study))   
        if len(citing_paper_dois) > 0:
            for paper_doi in citing_paper_dois:
                tasks.append(cq.get_citing_papers(paper_doi, citation_limit, from_dt, to_dt, fields_of_study))

        # --- Execute Concurrent Data Fetching Tasks (excluding expanded search for now) ---
        if tasks:
            logging.info(f"Running {len(tasks)} main data collection tasks concurrently...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results immediately
            for result in results:
                if isinstance(result, Exception):
                    logging.error(f"Data collection task failed: {result}")
                elif isinstance(result, list):
                    citation_results.extend(result) # Add items from successful tasks
                    logging.info("Main data collection tasks finished.")
        else:
            logging.info("No main data collection tasks to run.")

        # --- Add all aggregated items to the graph state ---
        self._add_items_json(citation_results, round, source="citation_expansion")
        logging.info(f"Graph state after collection tasks. Nodes: {len(self.nodes_json)}, Edges: {len(self.edges_json)}")
        
        # update explored nodes information
        if len(ref_paper_dois) > 0:
            self.explored_nodes['reference'].extend(ref_paper_dois) 
        if len(citing_paper_dois) > 0:
            self.explored_nodes['citing'].extend(citing_paper_dois) 


    async def author_expansion(
            self,
            seed_author_ids,
            candit_author_ids: List[str],
            if_include_seed_author: Optional[bool] = False,
            round: Optional[int] = 1,
            from_dt='2022-01-01',
            to_dt='2025-03-13',
            fields_of_study=None,
    ):
        """expand author information based on most cited authors
        - filter l authors who writes most papers in graph
        - search citations for these papers
        The citation for similar papers would help consolidate cross reference.
        """
        if not isinstance(candit_author_ids, list) or len(candit_author_ids) == 0:
            logging.warning("No candit authors from input.")
            author_results = []

        # whether exclude seed author
        if not if_include_seed_author:
            candit_author_ids = [x for x in candit_author_ids if x not in seed_author_ids]

        # exclude author that already with complete information
        expanded_author_ids = []
        for author_id in candit_author_ids:
            if author_id not in self.explored_nodes['author']:
                expanded_author_ids.append(author_id)

        # retrieve citation for top l most significant authors
        if isinstance(expanded_author_ids, list) and len(expanded_author_ids) > 0:
            logging.info(f"Preparing author info task for {len(expanded_author_ids)} authors.")
            aq = AuthorQuery()
            author_results = await aq.get_author_info(expanded_author_ids, from_dt, to_dt, fields_of_study)
        else:
            logging.info("No new authors found or all are already complete.")
            author_results = []

        # add item to nodes json or edges json
        self._add_items_json(author_results, round, source="author_expansion")

        # update explored nodes status
        if len(expanded_author_ids) > 0:
            self.explored_nodes['author'].extend(expanded_author_ids) 

