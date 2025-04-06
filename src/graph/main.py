import asyncio
import json
# import time # No longer needed for sleeps
import numpy as np
from collections import Counter
from typing import List, Dict, Optional, Union, Tuple # Added Tuple
from json_repair import repair_json

import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from utils.data_process import generate_hash_key
from apis.s2_api import SemanticScholarKit
from models.llms import async_llm_gen_w_retry
from prompts.query_prompt import keywords_topics_example, keywords_topics_prompt
from models.embedding_models import gemini_embedding_async, semantic_similarity_matrix # Assumed async where needed
from graph.s2_metadata_process import process_paper_metadata, process_citation_metadata, process_related_metadata, process_author_metadata # Assumed sync but fast

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define a default S2 API rate limit (requests per second)
# Semantic Scholar official limits: 100 requests per 5 minutes (~0.33 req/sec)
# Let's be conservative: 1 request every 4 seconds (0.25 req/sec)
S2_REQUEST_INTERVAL = 4 # Seconds between requests
S2_MAX_CONCURRENCE = 10 # max concurrence

class PaperSearch:
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
            min_citation_cnt: Optional[int] = None,
            institutions: Optional[List[str]] = None,
            journals: Optional[List[str]] = None,
            author_ids: Optional[List[str]] = None,
            s2_request_interval: float = S2_REQUEST_INTERVAL, # Rate limiting interval
            s2_max_concurrence: int = S2_MAX_CONCURRENCE,  # s2 max concurrence
    ):
        """
        Initialize PaperExploration parameters.
        (Args documentation omitted for brevity, same as original)
        """
        # CRITICAL: Ensure SemanticScholarKit is instantiated correctly (might need loop)
        self.s2 = SemanticScholarKit() # Ensure this doesn't block if it does setup
        self.research_topic = research_topic
        self.seed_paper_titles = [seed_paper_titles] if isinstance(seed_paper_titles, str) and seed_paper_titles else seed_paper_titles if isinstance(seed_paper_titles, list) else []
        self.seed_paper_dois = [seed_paper_dois] if isinstance(seed_paper_dois, str) and seed_paper_dois else seed_paper_dois if isinstance(seed_paper_dois, list) else []

        # Filters
        self.from_dt = from_dt
        self.to_dt = to_dt
        self.fields_of_study = fields_of_study
        self.min_citation_cnt = min_citation_cnt
        self.institutions = institutions
        self.journals = journals
        self.author_ids = author_ids

        # LLM/Embedding
        self.llm_api_key = llm_api_key
        self.llm_model_name = llm_model_name
        self.embed_api_key = embed_api_key
        self.embed_model_name = embed_model_name

        # State: Nodes and Edges
        self.nodes_json: List[Dict] = []
        self.edges_json: List[Dict] = []
        # Use a set for faster node/edge existence checks during addition
        self._node_ids = set()
        self._edge_tuples = set() # Store (start_id, end_id, type) tuples

        # --- Async specific ---
        # Semaphore for rate limiting S2 API calls
        # Allow 1 concurrent request, enforce delay between requests externally
        self._s2_max_concurrence = s2_max_concurrence
        self._s2_semaphore = asyncio.Semaphore(self._s2_max_concurrence)
        self._s2_request_interval = s2_request_interval
        self._last_s2_request_time = 0

    async def _rate_limited_s2_call(self, coro):
        """Wrapper to enforce rate limit before executing an S2 API coroutine."""
        async with self._s2_semaphore:
            current_time = asyncio.get_event_loop().time()
            time_since_last = current_time - self._last_s2_request_time
            if time_since_last < self._s2_request_interval:
                await asyncio.sleep(self._s2_request_interval - time_since_last)

            # Execute the actual S2 call
            result = await coro
            self._last_s2_request_time = asyncio.get_event_loop().time()
            return result

    def _add_items_to_graph(self, items: List[Dict]):
        """Adds nodes and relationships from processed data"""
        nodes_added = 0
        edges_added = 0
        for item in items:
            if item['type'] == 'node':
                node_id = item['id']
                self.nodes_json.append(item)
                self._node_ids.add(node_id)
                nodes_added += 1
            elif item['type'] == 'relationship':
                rel_type = item['relationshipType']
                start_id = item['startNodeId']
                end_id = item['endNodeId']
                edge_tuple = (start_id, end_id, rel_type)
                if edge_tuple not in self._edge_tuples:
                    self.edges_json.append(item)
                    self._edge_tuples.add(edge_tuple)
                    edges_added += 1
        # logging.debug(f"Added {nodes_added} nodes, {edges_added} edges.") # Optional debug log


    async def initial_paper_query(
            self,
            limit: Optional[int] = 100,
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
    ):
        """
        Retrieve papers based on user's input text asynchronously.
        (Args documentation omitted for brevity)
        """
        tasks = []
        fields_of_study = fields_of_study if fields_of_study is not None else self.fields_of_study
        from_dt = from_dt if from_dt is not None else self.from_dt
        to_dt = to_dt if to_dt is not None else self.to_dt

        # Task for searching by DOIs
        if self.seed_paper_dois:
            async def _fetch_by_dois():
                logging.info(f"Fetching papers by {len(self.seed_paper_dois)} DOIs...")
                s2_paper_metadata = await self._rate_limited_s2_call(
                    self.s2.async_search_paper_by_ids(id_list=self.seed_paper_dois)
                )
                logging.info(f"Processing {len(s2_paper_metadata)} papers from DOIs.")
                processed = process_paper_metadata(
                    s2_paper_metadata=s2_paper_metadata, 
                    from_dt=from_dt, 
                    to_dt=to_dt, 
                    fields_of_study=fields_of_study)
                # Mark as seed papers
                for item in processed:
                     if item['type'] == 'node' and item['labels'] == ['Paper']:
                         item['properties']['from_seed'] = True
                         item['properties']['is_complete'] = True # Assume initial fetch is complete
                return processed, [] # Return processed nodes/edges and empty searched list
            tasks.append(_fetch_by_dois())

        # Tasks for searching by Titles
        if self.seed_paper_titles:
            async def _fetch_by_title(title):
                logging.info(f"Fetching papers by title: '{title[:50]}...'")
                s2_paper_metadata = await self._rate_limited_s2_call(
                    self.s2.async_search_paper_by_keywords(query=title, fields_of_study=fields_of_study, limit=limit)
                )
                seed_meta, searched_meta = [], []
                if s2_paper_metadata:
                    seed_meta.append(s2_paper_metadata[0]) # Assume first is the seed match
                    searched_meta.extend(s2_paper_metadata[1:])
                logging.info(f"Processing {len(seed_meta)} seed and {len(searched_meta)} searched papers for title '{title[:50]}...'")
                processed_seed = process_paper_metadata(
                    s2_paper_metadata=seed_meta, 
                    from_dt=from_dt, 
                    to_dt=to_dt, 
                    fields_of_study=fields_of_study)
                processed_searched = process_paper_metadata(
                    s2_paper_metadata=searched_meta, 
                    from_dt=from_dt, 
                    to_dt=to_dt, 
                    fields_of_study=fields_of_study)
                # Mark appropriately
                for item in processed_seed:
                     if item['type'] == 'node' and item['labels'] == ['Paper']:
                         item['properties']['from_seed'] = True
                         item['properties']['from_title_search'] = True
                         item['properties']['is_complete'] = True
                for item in processed_searched:
                     if item['type'] == 'node' and item['labels'] == ['Paper']:
                         item['properties']['from_title_search'] = True
                         item['properties']['is_complete'] = True
                return processed_seed, processed_searched

            for title in self.seed_paper_titles:
                 tasks.append(_fetch_by_title(title))


        # Task for searching by Research Topic
        if self.research_topic:
            async def _fetch_by_topic():
                logging.info(f"Fetching papers by topic: '{self.research_topic[:50]}...'")
                s2_paper_metadata = await self._rate_limited_s2_call(
                    self.s2.async_search_paper_by_keywords(query=self.research_topic, fields_of_study=fields_of_study, limit=limit)
                )
                logging.info(f"Processing {len(s2_paper_metadata)} papers from topic search.")
                processed = process_paper_metadata(
                    s2_paper_metadata=s2_paper_metadata, 
                    from_dt=from_dt, 
                    to_dt=to_dt, 
                    fields_of_study=fields_of_study)
                # Mark as searched papers
                for item in processed:
                     if item['type'] == 'node' and item['labels'] == ['Paper']:
                         item['properties']['from_topic_search'] = True
                         item['properties']['is_complete'] = True
                return [], processed # Return empty seed list and processed searched list
            tasks.append(_fetch_by_topic())


        # Run all initial query tasks concurrently
        all_results = []
        if tasks:
            logging.info(f"Running {len(tasks)} initial query tasks concurrently...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            all_results.extend(results)
        else:
            logging.warning("No initial query criteria (DOI, Title, Topic) provided.")
            return # Exit if no query was made

        # Process results and add to graph state
        logging.info("Aggregating results from initial queries...")
        for result in all_results:
            if isinstance(result, Exception):
                logging.error(f"Initial query task failed: {result}")
            elif isinstance(result, tuple) and len(result) == 2:
                seed_items, searched_items = result
                self._add_items_to_graph(seed_items)
                self._add_items_to_graph(searched_items)
            else:
                logging.warning(f"Unexpected result type from initial query task: {type(result)}")

        # Update the internal set of known node IDs
        self._node_ids = {node['id'] for node in self.nodes_json}
        logging.info(f"Initial query complete. Nodes: {len(self.nodes_json)}, Edges: {len(self.edges_json)}")


    async def get_author_info(
            self,
            author_ids: List[str],
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
    ) -> List[Dict]: # Return processed items instead of modifying state directly
        """Fetches and processes author information asynchronously."""
        if not author_ids:
            return []

        fields_of_study = fields_of_study if fields_of_study is not None else self.fields_of_study
        from_dt = from_dt if from_dt is not None else self.from_dt
        to_dt = to_dt if to_dt is not None else self.to_dt

        logging.info(f"Fetching info for {len(author_ids)} authors...")
        authors_info = await self._rate_limited_s2_call(
            self.s2.async_search_author_by_ids(author_ids=author_ids, with_abstract=True) # Assuming fields apply here too
        )

        logging.info(f"Processing metadata for {len(authors_info)} authors.")
        s2_author_meta_json = process_author_metadata(
            s2_author_metadata=authors_info,
            from_dt=from_dt,
            to_dt=to_dt,
            fields_of_study=fields_of_study
        )

        # Mark nodes appropriately before returning
        for item in s2_author_meta_json:
            if item['type'] == 'node':
                if item['labels'] == ['Paper']:
                    item['properties']['from_same_author'] = True
                    item['properties']['is_complete'] = True # Assume author paper fetch is complete
                elif item['labels'] == ['Author']:
                    item['properties']['is_complete'] = True # Author node itself is complete
        return s2_author_meta_json


    async def get_cited_papers(
            self,
            paper_doi: str,
            limit: Optional[int] = 100,
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
    ) -> List[Dict]: # Return processed items
        """Get papers cited by the paper asynchronously."""
        fields_of_study = fields_of_study if fields_of_study is not None else self.fields_of_study
        from_dt = from_dt if from_dt is not None else self.from_dt
        to_dt = to_dt if to_dt is not None else self.to_dt

        logging.info(f"Fetching papers cited by {paper_doi}...")
        s2_citedpaper_metadata = await self._rate_limited_s2_call(
            self.s2.async_get_s2_cited_papers(paper_doi, limit=limit, with_abstract=True)
        )

        logging.info(f"Processing {len(s2_citedpaper_metadata)} cited papers for {paper_doi}.")
        s2_citedpapermeta_json = process_citation_metadata(
            original_paper_doi=paper_doi,
            s2_citation_metadata=s2_citedpaper_metadata,
            citation_type='citedPaper', # Reference (paper cites this)
            from_dt=from_dt,
            to_dt=to_dt,
            fields_of_study=fields_of_study
        )
        # Mark nodes
        for item in s2_citedpapermeta_json:
            if item['type'] == 'node' and item['labels'] == ['Paper']:
                item['properties']['from_reference'] = True # Changed from 'from_cited' for clarity (it's a reference)
                item['properties']['is_complete'] = True
        return s2_citedpapermeta_json


    async def get_citing_papers(
            self,
            paper_doi: str,
            limit: Optional[int] = 100,
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
    ) -> List[Dict]: # Return processed items
        """Retrieve papers that cite the paper asynchronously."""
        fields_of_study = fields_of_study if fields_of_study is not None else self.fields_of_study
        from_dt = from_dt if from_dt is not None else self.from_dt
        to_dt = to_dt if to_dt is not None else self.to_dt

        logging.info(f"Fetching papers citing {paper_doi}...")
        s2_citingpaper_metadata = await self._rate_limited_s2_call(
            self.s2.async_get_s2_citing_papers(paper_doi, limit=limit, with_abstract=True)
        )

        logging.info(f"Processing {len(s2_citingpaper_metadata)} citing papers for {paper_doi}.")
        s2_citingpapermetadata_json = process_citation_metadata(
            original_paper_doi=paper_doi,
            s2_citation_metadata=s2_citingpaper_metadata,
            citation_type='citingPaper', # Citation (this paper cites original)
            from_dt=from_dt,
            to_dt=to_dt,
            fields_of_study=fields_of_study
        )
        # Mark nodes
        for item in s2_citingpapermetadata_json:
            if item['type'] == 'node' and item['labels'] == ['Paper']:
                item['properties']['from_citation'] = True # Changed from 'from_citing'
                item['properties']['is_complete'] = True
        return s2_citingpapermetadata_json

    async def get_recommend_papers(
            self,
            paper_dois: List[str], # Expecting list now based on gather usage
            limit: Optional[int] = 100,
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
    ) -> List[Dict]: # Return processed items
        """Retrieve recommended papers asynchronously."""
        if not paper_dois: return []

        fields_of_study = fields_of_study if fields_of_study is not None else self.fields_of_study
        from_dt = from_dt if from_dt is not None else self.from_dt
        to_dt = to_dt if to_dt is not None else self.to_dt

        logging.info(f"Fetching recommendations based on {len(paper_dois)} papers...")
        s2_recommended_metadata = await self._rate_limited_s2_call(
            self.s2.async_get_s2_recommended_papers(positive_paper_ids=paper_dois, limit=limit, with_abstract=True)
        )

        logging.info(f"Processing {len(s2_recommended_metadata)} recommended papers.")
        s2_recpapermetadata_json = process_paper_metadata(
            s2_paper_metadata=s2_recommended_metadata,
            from_dt=from_dt,
            to_dt=to_dt,
            fields_of_study=fields_of_study
        )
        # Mark nodes
        for item in s2_recpapermetadata_json:
            if item['type'] == 'node' and item['labels'] == ['Paper']:
                item['properties']['from_recommended'] = True
                item['properties']['is_complete'] = True
        return s2_recpapermetadata_json


    async def llm_gen_related_topics(self, seed_paper_ids: List[str]) -> Optional[Dict]:
        """Generate related topics using LLM asynchronously."""
        logging.info(f"Generating related topics for {len(seed_paper_ids)} seed papers...")
        # Use existing node data, assuming initial query ran
        nodes_dict = {node['id']: node for node in self.nodes_json}
        domains, seed_paper_texts = [], []

        for paper_id in seed_paper_ids:
            item = nodes_dict.get(paper_id)
            if item and item['labels'] == ['Paper']:
                title = item['properties'].get('title')
                abstract = item['properties'].get('abstract')
                domain_list = item['properties'].get('fieldsOfStudy', []) # Default to empty list
                if title and abstract:
                    info = f"<paper> {title}\n{abstract} </paper>"
                    seed_paper_texts.append(info)
                if domain_list: # Check if list is not empty
                    domains.extend(domain_list)
            else:
                 logging.warning(f"Seed paper ID {paper_id} not found or not a Paper node.")

        if not seed_paper_texts:
            logging.error("No text found for seed papers to generate topics.")
            return None
        if not domains:
            logging.warning("No domains found for seed papers, using 'General Science'.")
            domain = "General Science"
        else:
            # Get the most frequent domain
            domain_counts = Counter(domains)
            if not domain_counts: # Handle empty counter if domains list was empty
                domain = "General Science"
            else:
                domain = domain_counts.most_common(1)[0][0]


        # LLM call (assumed async)
        qa_prompt = keywords_topics_prompt.format(
            domain=domain,
            example_json=keywords_topics_example,
            input_text="\n\n".join(seed_paper_texts)
        )
        logging.info("Calling LLM to generate topics...")
        try:
            # Assuming llm_gen_w_retry is async
            keywords_topics_info = await async_llm_gen_w_retry(self.llm_api_key, self.llm_model_name, qa_prompt, sys_prompt=None, temperature=0.6)
            if not keywords_topics_info:
                 logging.error("LLM returned empty response for topic generation.")
                 return None

            # Repair and parse JSON
            repaired_json_str = repair_json(keywords_topics_info)
            keywords_topics_json = json.loads(repaired_json_str)
            logging.info(f"LLM generated topics: {json.dumps(keywords_topics_json)}")

        except json.JSONDecodeError as e:
            logging.error(f"LLM Topic Generation - JSON Repair/Decode Error: {e}. Original output: {keywords_topics_info}")
            return None
        except Exception as e:
            logging.error(f"Error during LLM topic generation or processing: {e}")
            return None


        # Add Topic nodes and relationships (synchronous part, happens after await)
        processed_topic_items = []
        query_topics = keywords_topics_json.get('queries', []) # Default to empty list
        current_node_ids = {node['id'] for node in self.nodes_json} # Get current node IDs

        for topic in query_topics:
            topic_hash_id = generate_hash_key(topic)
            if topic_hash_id not in current_node_ids:
                topic_node = {
                    'type': 'node',
                    'id': topic_hash_id,
                    'labels': ['Topic'],
                    'properties': {'topic_hash_id': topic_hash_id, 'topic_name': topic, 'hash_method': 'hashlib.sha256'}
                }
                processed_topic_items.append(topic_node)
                current_node_ids.add(topic_hash_id) # Add locally to avoid adding duplicates within this loop

            for paper_id in seed_paper_ids:
                 if paper_id in nodes_dict: # Check if seed paper exists
                    # Check if edge already exists using the internal set for efficiency
                    edge_tuple = (paper_id, topic_hash_id, "DISCUSS")
                    if edge_tuple not in self._edge_tuples:
                        paper_topic_relationship = {
                            "type": "relationship",
                            "relationshipType": "DISCUSS",
                            "startNodeId": paper_id,
                            "endNodeId": topic_hash_id,
                            "properties": {}
                        }
                        processed_topic_items.append(paper_topic_relationship)
                        self._edge_tuples.add(edge_tuple) # Add locally


        # Add newly created topic nodes/edges to the main graph state
        self._add_items_to_graph(processed_topic_items)
        logging.info(f"Added {len(processed_topic_items)} topic nodes/edges.")

        return keywords_topics_json


    async def get_related_papers(
            self,
            topic_json: Dict,
            limit: Optional[int] = 100,
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
    ) -> List[Dict]: # Return aggregated processed items
        """Fetch related papers based on LLM queries concurrently."""
        queries = topic_json.get('queries', [])
        if not queries:
            logging.warning("No queries found in topic_json for related paper search.")
            return []

        fields_of_study = fields_of_study if fields_of_study is not None else self.fields_of_study
        from_dt = from_dt if from_dt is not None else self.from_dt
        to_dt = to_dt if to_dt is not None else self.to_dt

        tasks = []

        async def _fetch_related_for_query(query):
            logging.info(f"Fetching related papers for query: '{query[:50]}...'")
            s2_paper_metadata = await self._rate_limited_s2_call(
                self.s2.async_search_paper_by_keywords(query, fields_of_study=fields_of_study, limit=limit)
            )
            logging.info(f"Processing {len(s2_paper_metadata)} related papers for query '{query[:50]}...'")
            s2_papermeta_json = process_related_metadata( 
                s2_related_metadata=s2_paper_metadata,
                topic=query,
                from_dt=from_dt,
                to_dt=to_dt,
                fields_of_study=fields_of_study
            )
            # Mark nodes
            for item in s2_papermeta_json:
                if item['type'] == 'node' and item['labels'] == ['Paper']:
                    item['properties']['from_related_topics'] = True
                    item['properties']['is_complete'] = True
            return s2_papermeta_json

        for query in queries:
            tasks.append(_fetch_related_for_query(query))

        # Run all related paper searches concurrently
        all_processed_items = []
        if tasks:
            logging.info(f"Running {len(tasks)} related paper search tasks concurrently...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logging.error(f"Related paper search task failed: {result}")
                elif isinstance(result, list):
                    all_processed_items.extend(result)
        return all_processed_items


    async def cal_semantic_similarity(self, paper_nodes_json: List[Dict]) -> List[Dict]:
        """Calculate semantic similarity asynchronously."""
        # (Code is largely the same as original, assuming gemini_embedding_async is async
        # and semantic_similarity_matrix is either fast sync or wrapped if needed)
        semantic_similar_pool = []
        publish_dt_ref = {x['id']:x['properties'].get('publicationDate')
                           for x in paper_nodes_json if x['properties'].get('publicationDate') is not None}

        ids, texts = [], []
        for node in paper_nodes_json:
            node_id = node['id'] # Renamed variable
            title = node['properties'].get('title')
            abstract = node['properties'].get('abstract')
            if title is not None and abstract is not None:
                texts.append(f"TITLE: {title} \nABSTRACT: {abstract}")
                ids.append(node_id)

        if not texts:
            logging.warning("No paper titles and abstracts found for semantic similarity calculation.")
            return []

        paper_info_ref = {key: value for key, value in zip(ids, texts)} # Build ref after filtering

        logging.info(f"Generating embeddings for {len(texts)} papers...")
        # Assume gemini_embedding_async handles its own batching/rate limits if necessary
        try:
             embeds = await gemini_embedding_async(self.embed_api_key, self.embed_model_name, texts, 10) # Pass batch size
        except Exception as e:
             logging.error(f"Embedding generation failed: {e}")
             return []

        if embeds is None or (len(embeds) != len(texts)):
             logging.error(f"Embedding generation returned unexpected result. Expected {len(texts)} embeddings, got {len(embeds) if embeds else 0}.")
             return []

        logging.info("Calculating similarity matrix...")
        try:
            sim_matrix = semantic_similarity_matrix(embeds, embeds)
            sim_matrix = np.array(sim_matrix)
        except Exception as e:
            logging.error(f"Similarity matrix calculation failed: {e}")
            return []

        logging.info("Processing similarity matrix to create relationships...")
        rows, cols = sim_matrix.shape
        # Use a set to avoid duplicate relationships (A->B vs B->A with same score)
        added_pairs = set()

        for i in range(rows):
            id_i = ids[i]
            publish_dt_i = publish_dt_ref.get(id_i) 
            for j in range(i + 1, cols): # Iterate only upper triangle (j > i)
                id_j = ids[j]
                sim = sim_matrix[i, j]
                publish_dt_j = publish_dt_ref.get(id_j) # Not needed
                
                if publish_dt_i <= publish_dt_j:
                     start_node_id = id_i
                     end_node_id = id_j
                else:
                     start_node_id = id_j
                     end_node_id = id_i

                # Check if this pair (regardless of direction initially computed) was added
                pair_tuple = (start_node_id, end_node_id)
                if pair_tuple not in added_pairs:
                    edge = {
                        "type": "relationship",
                        "relationshipType": "SIMILAR_TO",
                        "startNodeId": start_node_id,
                        "endNodeId": end_node_id,
                        "properties": {
                             'source': 'semantic similarity',
                             'weight': round(float(sim), 4), # Ensure float conversion from numpy type
                         }
                     }
                    semantic_similar_pool.append(edge)
                    added_pairs.add(pair_tuple)

        logging.info(f"Generated {len(semantic_similar_pool)} potential similarity relationships.")
        return semantic_similar_pool


    # init_collect remains synchronous as it calls initial_paper_query which is now async
    # It should likely be async itself if called from an async context.
    async def init_collect(
            self,
            limit=50,
            from_dt='2022-01-01',
            to_dt='2025-03-13',
            fields_of_study=None,
    ):
        """Initializes the collection by running the initial paper query."""
        await self.initial_paper_query(limit=limit, from_dt=from_dt, to_dt=to_dt, fields_of_study=fields_of_study)


    # The main optimized async collect method
    async def collect(
            self,
            seed_paper_dois: List[str], # Use the DOIs provided here
            with_reference: Optional[bool] = True,
            with_author: Optional[bool] = True,
            with_recommend: Optional[bool] = True,
            with_expanded_search: Optional[bool] = True,
            add_semantic_similarity: Optional[bool] = True,
            similarity_threshold: Optional[float] = 0.7,
            limit=50,
            from_dt=from_dt,
            to_dt='2025-03-13',
            fields_of_study=None,
    ):
        """
        Main asynchronous collection method orchestrating various data fetching tasks.
        """
        # Ensure seed papers are loaded if not already (or rely on init_collect being called first)
        # Optional: Check if seed_paper_dois exist in self.nodes_json, maybe run parts of initial_paper_query if not?
        # For simplicity, assume init_collect or equivalent setup has happened.
        if not seed_paper_dois:
             logging.warning("Collect called with no seed paper DOIs.")
             # Decide behaviour: exit? or proceed with other tasks if possible?
             # Let's assume we proceed based on existing graph if seed_paper_dois is empty but flags are True

        tasks = []
        processed_items_aggregate = [] # Store results from tasks

        # --- Task Prep: Author Info ---
        if with_author:
            # Find authors of the *provided* seed_paper_dois from current graph state
            author_ids_to_fetch = set()
            current_nodes_dict = {node['id']: node for node in self.nodes_json}
            current_complete_author_ids = {node['id'] for node in self.nodes_json if node['labels'] == ["Author"] and node['properties'].get('is_complete')}

            for doi in seed_paper_dois:
                 paper_node = current_nodes_dict.get(doi)
                 if paper_node and paper_node['labels'] == ["Paper"]:
                     authors = paper_node['properties'].get('authors', [])
                     for author in authors:
                         author_id = author.get('authorId')
                         if author_id and author_id not in current_complete_author_ids:
                             author_ids_to_fetch.add(author_id)

            if author_ids_to_fetch:
                 logging.info(f"Preparing author info task for {len(author_ids_to_fetch)} authors.")
                 tasks.append(self.get_author_info(list(author_ids_to_fetch), from_dt, to_dt, fields_of_study))
            else:
                 logging.info("No new authors found for seed DOIs or all are already complete.")


        # --- Task Prep: References & Citations ---
        if with_reference and seed_paper_dois:
            logging.info(f"Preparing reference/citation tasks for {len(seed_paper_dois)} seed papers.")
            for paper_doi in seed_paper_dois:
                tasks.append(self.get_cited_papers(paper_doi, limit, from_dt, to_dt, fields_of_study))
                tasks.append(self.get_citing_papers(paper_doi, limit, from_dt, to_dt, fields_of_study))

        # --- Task Prep: Recommendations ---
        if with_recommend and seed_paper_dois:
             logging.info("Preparing recommendations task.")
             tasks.append(self.get_recommend_papers(seed_paper_dois, limit, from_dt, to_dt, fields_of_study))


        # --- Task Prep: Expanded Search ---
        # Needs to run sequentially internally (LLM -> Search) but concurrently with others
        expanded_search_task = None
        if with_expanded_search and seed_paper_dois:
            async def _handle_expanded_search():
                logging.info("Starting expanded search sub-task...")
                keywords_topics_json = await self.llm_gen_related_topics(seed_paper_dois)
                print(keywords_topics_json)
                related_papers_items = []
                if keywords_topics_json:
                    # Now fetch related papers based on the generated topics
                    related_papers_items = await self.get_related_papers(keywords_topics_json, limit, from_dt, to_dt, fields_of_study)
                else:
                    logging.warning("Skipping related paper fetch as no topics were generated.")
                # Topic nodes/edges were already added within llm_gen_related_topics
                logging.info(f"Exapned topic and get {len(related_papers_items)} more articles")
                return related_papers_items # Return only the newly fetched paper items

            logging.info("Preparing expanded search task.")
            expanded_search_task = asyncio.create_task(_handle_expanded_search()) # Create task immediately


        # --- Execute Concurrent Data Fetching Tasks (excluding expanded search for now) ---
        if tasks:
            logging.info(f"Running {len(tasks)} main data collection tasks concurrently...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            logging.info("Main data collection tasks finished.")
            # Process results immediately
            for result in results:
                if isinstance(result, Exception):
                    logging.error(f"Data collection task failed: {result}")
                elif isinstance(result, list):
                    processed_items_aggregate.extend(result) # Add items from successful tasks
        else:
            logging.info("No main data collection tasks to run.")


        # --- Wait for and Process Expanded Search Task ---
        if expanded_search_task:
            logging.info("Waiting for expanded search task to complete...")
            try:
                expanded_result = await expanded_search_task
                if isinstance(expanded_result, list):
                     processed_items_aggregate.extend(expanded_result)
                logging.info("Expanded search task finished.")
            except Exception as e:
                 logging.error(f"Expanded search task failed: {e}")


        # --- Add all aggregated items to the graph state ---
        logging.info(f"Adding {len(processed_items_aggregate)} items from all tasks to graph...")
        self._add_items_to_graph(processed_items_aggregate)
        logging.info(f"Graph state after collection tasks. Nodes: {len(self.nodes_json)}, Edges: {len(self.edges_json)}")


        # --- Task: Semantic Similarity (after all nodes are potentially added) ---
        if add_semantic_similarity:
            logging.info("Starting semantic similarity calculation...")
            # Get all current paper nodes
            paper_nodes_for_similarity = [node for node in self.nodes_json if node['labels'] == ["Paper"]]
            if paper_nodes_for_similarity:
                try:
                    semantic_similar_pool = await self.cal_semantic_similarity(paper_nodes_for_similarity)
                    # Filter by threshold and add to graph
                    semantic_similar_relationships = [
                        edge for edge in semantic_similar_pool
                        if edge['properties'].get('weight', 0) > similarity_threshold
                    ]
                    logging.info(f"Adding {len(semantic_similar_relationships)} similarity edges above threshold {similarity_threshold}...")
                    self._add_items_to_graph(semantic_similar_relationships) # Add filtered edges
                except Exception as e:
                    logging.error(f"Semantic similarity calculation or addition failed: {e}")
            else:
                 logging.warning("No paper nodes found to calculate semantic similarity.")

        # Final state log
        logging.info(f"Collection process complete. Final state: Nodes={len(self.nodes_json)}, Edges={len(self.edges_json)}")


# --- Example Usage ---
import os
async def paper_exploration(
        rounds: int = 2,
        research_topic: Optional[str]=None,
        seed_paper_dois: Optional[List[str]]=None,
        seed_paper_titles: Optional[List[str]]=None,
        with_reference: Optional[bool]=True,
        with_author: Optional[bool]=True, # Fetch authors of seed paper
        with_recommend: Optional[bool]=True,
        with_expanded_search: Optional[bool]=True, # Set to True to test LLM topic generation
        add_semantic_similarity: Optional[bool]=True,
        similarity_threshold: Optional[bool]=0.7,
        search_limit: Optional[int] = 50,
        recommend_limit: Optional[int] = 50,
        citation_limit: Optional[int] = 100,
        from_dt: Optional[str]="2000-01-01",
        to_dt: Optional[str]="9999-12-31"
        ):
    
    # Initialize with a seed DOI
    searcher = PaperSearch(
        research_topic=research_topic,
        seed_paper_dois=seed_paper_dois,
        seed_paper_titles=seed_paper_titles,
        llm_api_key=llm_api_key,
        llm_model_name=llm_model_name,
        embed_api_key=embed_api_key,
        embed_model_name=embed_model_name,
    )

    # Round 1. Run initial query to get seed paper info
    print("--- Running Initial Query ---")
    await searcher.init_collect(limit=search_limit, from_dt="2015-01-01", to_dt="2025-04-01")
    print(f"Nodes after init: {len(searcher.nodes_json)}")
    print(f"Edges after init: {len(searcher.edges_json)}")

    # Get the actual seed DOIs found (in case input was title/topic)
    # For this example, we know the input DOI, but in general:
    seed_dois_in_graph = [node['id'] for node in searcher.nodes_json if node['labels'] == ['Paper'] and node['properties'].get('from_seed')]
    if not seed_dois_in_graph:
         print("Warning: Seed DOI(s) not found after initial query.")
         return

    print(f"\n--- Running Main Collection for seed DOIs: {seed_dois_in_graph} ---")
    start_time = asyncio.get_event_loop().time()
    await searcher.collect(
        seed_paper_dois=seed_dois_in_graph, 
        with_reference=True,
        with_author=True, # Fetch authors of seed paper
        with_recommend=True,
        with_expanded_search=True, 
        add_semantic_similarity=True,
        similarity_threshold=0.6,
        limit=100, 
        from_dt=from_dt,
        to_dt=to_dt
    )
    end_time = asyncio.get_event_loop().time()
    print(f"\nOptimized collection took {end_time - start_time:.2f} seconds.")
    print(f"Final Nodes: {len(searcher.nodes_json)}")
    print(f"Final Edges: {len(searcher.edges_json)}")

    # Round 2. Run initial query to get seed paper info
    seed_dois_in_graph

    print("--- Running Initial Query ---")
    await searcher.init_collect(limit=50, from_dt="2015-01-01", to_dt="2025-04-01")
    print(f"Nodes after init: {len(searcher.nodes_json)}")
    print(f"Edges after init: {len(searcher.edges_json)}")

    # Get the actual seed DOIs found (in case input was title/topic)
    # For this example, we know the input DOI, but in general:
    seed_dois_in_graph = [node['id'] for node in searcher.nodes_json if node['labels'] == ['Paper'] and node['properties'].get('from_seed')]
    if not seed_dois_in_graph:
         print("Warning: Seed DOI(s) not found after initial query.")
         # Handle error or exit
         return

    print(f"\n--- Running Main Collection for seed DOIs: {seed_dois_in_graph} ---")
    start_time = asyncio.get_event_loop().time()
    await searcher.collect(
        seed_paper_dois=seed_dois_in_graph, 
        with_reference=True,
        with_author=True, # Fetch authors of seed paper
        with_recommend=True,
        with_expanded_search=True, # Set to True to test LLM topic generation
        add_semantic_similarity=True,
        similarity_threshold=0.6,
        limit=100, # Limit number of items per fetch step
        from_dt="2020-01-01",
        to_dt="2025-04-01"
    )
    end_time = asyncio.get_event_loop().time()
    print(f"\nOptimized collection took {end_time - start_time:.2f} seconds.")
    print(f"Final Nodes: {len(searcher.nodes_json)}")
    print(f"Final Edges: {len(searcher.edges_json)}")

    # Optional: Save or inspect searcher.nodes_json and searcher.edges_json
    with open("nodes.json", "w") as f:
        json.dump(searcher.nodes_json, f, indent=2)
    with open("edges.json", "w") as f:
        json.dump(searcher.edges_json, f, indent=2)

if __name__ == "__main__":

    # model setup
    llm_api_key = os.getenv('GEMINI_API_KEY_3')
    llm_model_name="gemini-2.0-flash"
    embed_api_key = os.getenv('GEMINI_API_KEY_3')
    embed_model_name="models/text-embedding-004"

    # initial seeds
    research_topic = "llm literature review"
    seed_dois = ['10.48550/arXiv.2406.10252',  # AutoSurvey: Large Language Models Can Automatically Write Surveys
                '10.48550/arXiv.2412.10415',  # Generative Adversarial Reviews: When LLMs Become the Critic
                '10.48550/arXiv.2402.12928',  # A Literature Review of Literature Reviews in Pattern Analysis and Machine Intelligence 
                ]
    seed_titles = ['PaperRobot: Incremental Draft Generation of Scientific Ideas',
                'From Hypothesis to Publication: A Comprehensive Survey of AI-Driven Research Support Systems'
                ]
    # Make sure to replace API keys before running
    # Configure logging level if needed
    # logging.getLogger().setLevel(logging.DEBUG) # For more verbose output
    asyncio.run(main())

