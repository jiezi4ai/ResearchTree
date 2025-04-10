import asyncio
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Literal # Added Tuple

import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from models.embedding_models import gemini_embedding_async, semantic_similarity_matrix

from graph.paper_graph import PaperGraph
from collect.paper_query import PaperQuery
from collect.citation_query import CitationQuery
from collect.author_query import AuthorQuery
from collect.paper_recommendation import PaperRecommendation
from collect.related_topic_query import RelatedTopicQuery

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PaperCollector:
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

        # LLM/Embedding
        self.llm_api_key = llm_api_key
        self.llm_model_name = llm_model_name
        self.embed_api_key = embed_api_key
        self.embed_model_name = embed_model_name

        # State: Nodes and Edges
        self.nodes_json = nodes_json if nodes_json else []
        self.edges_json = edges_json if edges_json else []
        # Use a set for faster node/edge existence checks during addition
        self._node_ids = set()
        self._edge_tuples = set() # Store (start_id, end_id, type) tuples

        # store paper abstraction embeddings
        self.abs_embed_ref = {}

        # paper graph
        self.pg = PaperGraph(name=paper_graph_name)
        if len(self.nodes_json) > 0:
            self.pg.add_graph_nodes(self.nodes_json)
        if len(self.edges_json) > 0:
            self.pg.add_graph_edges(self.edges_json)

        # explored nodes
        self.explored_nodes = {'seed':[],  # seed papers
                               'author':[],   # authors explored 
                               'reference':[],  # papers with reference explored
                               'citing': []  # papers with citation explored
                                }


    ############################################################################
    # basic function
    ############################################################################
    def _add_items_to_graph(self, items: List[Dict]):
        """Adds nodes and relationships from processed data
        """
        nodes_added, edges_added = 0, 0
        nodes_to_add, edges_to_add = [], []

        _node_ids = set()
        for x in self.nodes_json:
            # for author and paper, avoid duplication if there is complete information
            if x['labels'][0] in ['Author', 'Paper']:
                if x.get('properties', {}).get('is_complete') == True:
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
                    self.nodes_json.append(item)
                    nodes_added += 1
                    nodes_to_add.append(item)

                    if item['labels'][0] in ['Author', 'Paper']:
                        if item.get('properties', {}).get('is_complete') == True:
                            _node_ids.add(item['id'])
                    else:
                        _node_ids.add(item['id'])


            elif item['type'] == 'relationship':
                rel_type = item['relationshipType']
                start_id = item['startNodeId']
                end_id = item['endNodeId']
                edge_tuple = (start_id, end_id, rel_type) 
                if edge_tuple not in _edge_tuples:
                    self.edges_json.append(item)
                    _edge_tuples.add(edge_tuple)
                    edges_added += 1
                    edges_to_add.append(item)
            
        self.pg.add_graph_nodes(nodes_to_add)
        self.pg.add_graph_edges(edges_to_add)

 
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
    async def init_search(
            self,
            research_topic: Optional[List]=None,
            seed_paper_titles: Optional[List]=None,
            seed_paper_dois: Optional[List]=None,
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
        self._add_items_to_graph(initial_papers_json)
        logging.info(f"Graph state after initial search tasks. Nodes: {len(self.nodes_json)}, Edges: {len(self.edges_json)}")


    async def collect(
            self,
            seed_paper_dois: Optional[List[str]] = None,
            seed_author_ids: Optional[List[str]] = None,
            search_citation: Literal['reference', 'citing', 'both'] = None,
            search_author: Optional[bool] = False,
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
        self._add_items_to_graph(processed_items_aggregate)
        logging.info(f"Graph state after collection tasks. Nodes: {len(self.nodes_json)}, Edges: {len(self.edges_json)}")


    async def expand(
            self,
            seed_paper_json: Optional[List[Dict]],
            llm_api_key: Optional[str] = None,
            llm_model_name: Optional[str] = None,
            search_limit: Optional[int] = 50,
            from_dt='2022-01-01',
            to_dt='2025-03-13',
            fields_of_study=None,
    ):
        tq = RelatedTopicQuery(llm_api_key=llm_api_key, llm_model_name=llm_model_name)
        async def _handle_expanded_search():
            logging.info("Starting expanded search sub-task...")
            keywords_topics_json = await tq.llm_gen_related_topics(seed_paper_json)
            print(keywords_topics_json)

            seed_paper_topic_json = tq.get_topic_json(keywords_topics_json, seed_paper_json)

            related_papers_items = []
            if keywords_topics_json:
                # Now fetch related papers based on the generated topics
                try:
                    related_papers_items = await tq.get_related_papers(keywords_topics_json, search_limit, from_dt, to_dt, fields_of_study)
                except Exception as e:
                    logging.error(f"Failed to get related topicL {e}.")
            else:
                logging.warning("Skipping related paper fetch as no topics were generated.")

            expaned_items_json = seed_paper_topic_json + related_papers_items
            return expaned_items_json

        logging.info("Waiting for expanded search task to complete...")
        try:
            expanded_result = await asyncio.create_task(_handle_expanded_search())
            logging.info("Expanded search task finished.")
        except Exception as e:
            logging.error(f"Expanded search task failed: {e}")
            expanded_result = []

        # --- Add all aggregated items to the graph state ---
        self._add_items_to_graph(expanded_result)
        logging.info(f"Graph state after expansion tasks. Nodes: {len(self.nodes_json)}, Edges: {len(self.edges_json)}")


    ############################################################################
    # consolidation
    ############################################################################
    async def construct_paper_graph(
            self,
            search_citation: Literal['reference', 'citing', 'both'] = None,
            search_author: Optional[bool] = False,
            find_recommend: Optional[bool] = False,
            if_related_topic: Optional[bool] = False,
            if_expanded_citations: Literal['reference', 'citing', 'both'] = None,
            if_expanded_authors: Optional[bool] = False,
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
        await self.collect(
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
        if if_related_topic:
            await self.expand(
                seed_paper_json=seed_paper_json, 
                llm_api_key=self.llm_api_key, 
                llm_model_name=self.llm_model_name,
                search_limit=self.search_limit,
                from_dt=self.from_dt,
                to_dt=self.to_dt,
                fields_of_study = self.fields_of_study,
                )

        # --- CHECK CROSSREF PAPERS ---
        # get most similar papers to seed papers
        # track citation chain of these papers
        if if_expanded_citations:
            print(f"\n--- Get crossref papers: ---")
            non_seed_paper_dois = [node['id'] for node in self.nodes_json 
                                if node['labels'] == ['Paper'] and 
                                node['id'] not in seed_paper_dois]
            paper_nodes_json = [node for node in self.nodes_json 
                                if node['labels'] == ['Paper'] and 
                                node['properties'].get('title') is not None and node['properties'].get('abstract') is not None]
            
            # calculate paper nodes similarity
            semantic_similar_pool = await self.cal_embed_and_similarity(
                paper_nodes_json=paper_nodes_json,
                paper_dois_1=seed_paper_dois, 
                paper_dois_2=non_seed_paper_dois,
                similarity_threshold=similarity_threshold,
            )
            
            if len(semantic_similar_pool) > 0:
                candit_items = []
                for item in semantic_similar_pool:
                    wt = item.get('properties', {}).get('weight')
                    if (wt > 0.7 and wt < 0.95) or (wt > 0.7 and wt < 0.95):
                        if item['startNodeId'] in seed_paper_dois and item['endNodeId'] not in seed_paper_dois:
                            candit_items.append((item['endNodeId'], wt))
                        elif item['startNodeId'] not in seed_paper_dois and item['endNodeId'] in seed_paper_dois:
                            candit_items.append((item['startNodeId'], wt))
                sorted_items = sorted(candit_items, key=lambda item: item[1], reverse=True)

                # filter top k similarities
                expanded_paper_dois = [x[0] for x in sorted_items[0:expanded_k_papers]]
                print(f"Top similar papers are: {expanded_paper_dois}")

                # retrieve citation for top k similar papers
                await self.collect(
                    seed_paper_dois=expanded_paper_dois,
                    search_citation = if_expanded_citations,
                    citation_limit = self.citation_limit,
                    from_dt=self.from_dt,
                    to_dt=self.to_dt,
                    fields_of_study = self.fields_of_study,
                    )
                if search_citation in ['reference', 'both']:
                    self.explored_nodes['reference'].extend(expanded_paper_dois) 
                if search_citation in ['citing', 'both']:
                    self.explored_nodes['citing'].extend(expanded_paper_dois) 


        # --- CHECK SIGNIFICANT AUTHORS ---
        # filter most refered papers from graph
        # then search for author information
        if if_expanded_authors:
            print(f"\n--- Get most cited authors: ---")
            g = self.pg.graph
            author_ids = [node_id for node_id in g.nodes if g.nodes[node_id].get('nodeType') == 'Author'] 
            sorted_items = self.pg.cal_node_centrality(type_of_centrality='out')  # out refers to writes
            expanded_author_ids = [x[0] for x in sorted_items if x[0] in author_ids and x[0] not in seed_author_ids][0:expanded_l_authors]
            print(f"Get {len(expanded_author_ids)} authors for further exploration")

            # retrieve citation for top l most significant authors
            await self.collect(
                seed_author_ids=expanded_author_ids,
                search_author = if_expanded_authors,
                from_dt=self.from_dt,
                to_dt=self.to_dt,
                fields_of_study = self.fields_of_study,
                )
            self.explored_nodes['author'].extend(expanded_author_ids) 
                
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
