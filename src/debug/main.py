import asyncio
import json
import numpy as np
from collections import Counter
from typing import List, Dict, Optional, Union, Tuple, Any
from json_repair import repair_json
from aiolimiter import AsyncLimiter

from utils.data_process import generate_hash_key
from s2_api import SemanticScholarKit
from llms import async_llm_gen_w_retry
from prompts.query_prompt import keywords_topics_example, keywords_topics_prompt
from models.embedding_models import gemini_embedding_async, semantic_similarity_matrix
from graph.s2_metadata_process import process_paper_metadata, process_citation_metadata, process_related_metadata, process_author_metadata

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Default S2 API rate limit guidance: 100 requests per 5 minutes (~0.33 req/sec)
# Default interval ensures roughly 1 request every 4 seconds (0.25 req/sec)
DEFAULT_S2_REQUEST_INTERVAL = 4
# Default max concurrent requests allowed to S2 API
DEFAULT_S2_MAX_CONCURRENT = 1

class PaperSearch:
    """
    A class for exploring academic papers using Semantic Scholar API and LLMs,
    optimized for asynchronous operations with controlled concurrency and rate limiting.
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
            min_citation_cnt: Optional[int] = 0,
            institutions: Optional[List[str]] = None,
            journals: Optional[List[str]] = None,
            author_ids: Optional[List[str]] = None,
            # --- S2 API Control Parameters ---
            s2_request_interval: float = DEFAULT_S2_REQUEST_INTERVAL, # Avg seconds between requests
            s2_max_concurrent: int = DEFAULT_S2_MAX_CONCURRENT, # Max simultaneous requests
    ):
        """
        Initialize PaperExploration parameters.
        (Args documentation omitted for brevity, same as original)
        New Args:
            s2_request_interval (float): Target average interval (seconds) between S2 API requests for rate limiting. Defaults to 4.
            s2_max_concurrent (int): Maximum number of concurrent requests allowed to the S2 API. Defaults to 5.
        """
        # set up SemanticScholarKit
        self.s2 = SemanticScholarKit()
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

        # LLM/Embedding (Using async versions now)
        self.llm_api_key = llm_api_key
        self.llm_model_name = llm_model_name
        self.embed_api_key = embed_api_key
        self.embed_model_name = embed_model_name

        # State: Nodes and Edges
        self.nodes_json: List[Dict] = []
        self.edges_json: List[Dict] = []
        self._node_ids = set()
        self._edge_tuples = set()

        # --- Async specific ---
        if s2_max_concurrent <= 0:
            raise ValueError("s2_max_concurrent must be positive")
        if s2_request_interval <= 0:
             raise ValueError("s2_request_interval must be positive")

        # Semaphore to limit MAX CONCURRENT requests
        self._s2_semaphore = asyncio.Semaphore(s2_max_concurrent)

        # Rate limiter (aiolimiter) to control requests PER SECOND on average
        # Rate = 1 request per s2_request_interval seconds
        rate_limit = 1.0 / s2_request_interval
        self._s2_rate_limiter = AsyncLimiter(rate_limit, 1.0) # Max rate, per 1 second

        logging.info(f"S2 API Control: Max Concurrent={s2_max_concurrent}, Avg Interval={s2_request_interval:.2f}s (Rate Limit ~{rate_limit:.2f}/sec)")


    async def _s2_api_call_wrapper(self, coro):
        """
        Wrapper for S2 API calls to enforce both concurrency limit and rate limit.
        Acquires rate limiter first, then concurrency semaphore.
        """
        async with self._s2_rate_limiter: # Wait until allowed by rate limit
            async with self._s2_semaphore: # Wait for an available concurrency slot
                # Once both are acquired, execute the actual S2 call coroutine
                # No manual sleep needed here.
                result = await coro
                return result
        # Semaphore and rate limit allowance are released automatically upon exiting 'async with' blocks

    def _add_items_to_graph(self, items: List[Dict]):
        """Adds nodes and relationships from processed data, avoiding duplicates."""
        nodes_added = 0
        edges_added = 0
        for item in items:
            if not isinstance(item, dict): # Basic check
                 logging.warning(f"Skipping non-dict item during graph update: {type(item)}")
                 continue

            item_type = item.get('type')
            if item_type == 'node':
                node_id = item.get('id')
                if node_id and node_id not in self._node_ids:
                    self.nodes_json.append(item)
                    self._node_ids.add(node_id)
                    nodes_added += 1
            elif item_type == 'relationship':
                rel_type = item.get('relationshipType')
                start_id = item.get('startNodeId')
                end_id = item.get('endNodeId')
                if rel_type and start_id and end_id:
                    # Use a canonical tuple (sorted IDs for undirected like SIMILAR_TO)
                    # Keep order for directed (e.g., CITES, AUTHOR_OF)
                    # Assuming SIMILAR_TO is undirected for canonical check
                    if rel_type == "SIMILAR_TO":
                         edge_tuple_key = tuple(sorted((start_id, end_id))) + (rel_type,)
                    else:
                         edge_tuple_key = (start_id, end_id, rel_type)

                    if edge_tuple_key not in self._edge_tuples:
                        self.edges_json.append(item)
                        self._edge_tuples.add(edge_tuple_key)
                        edges_added += 1
            # else: ignore items without 'type' or with unknown 'type'

        if nodes_added > 0 or edges_added > 0:
             logging.debug(f"Added {nodes_added} nodes, {edges_added} edges.")


    async def initial_paper_query(
            self,
            limit: Optional[int] = 100,
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
    ):
        """
        Retrieve papers based on user's input text asynchronously.
        (Uses the _s2_api_call_wrapper for concurrency/rate limits)
        """
        tasks = []
        fields_of_study = fields_of_study if fields_of_study is not None else self.fields_of_study
        from_dt = from_dt if from_dt is not None else self.from_dt
        to_dt = to_dt if to_dt is not None else self.to_dt

        # Task for searching by DOIs
        if self.seed_paper_dois:
            async def _fetch_by_dois():
                logging.info(f"Fetching papers by {len(self.seed_paper_dois)} DOIs...")
                # Use the wrapper for the S2 call
                s2_paper_metadata = await self._s2_api_call_wrapper(
                    self.s2.search_paper_by_ids(id_list=self.seed_paper_dois)
                )
                logging.info(f"Processing {len(s2_paper_metadata)} papers from DOIs.")
                processed = process_paper_metadata(s2_paper_metadata, from_dt, to_dt, fields_of_study)
                for item in processed:
                     if item.get('type') == 'node' and item.get('labels') == ['Paper']:
                         item.setdefault('properties', {})['from_seed'] = True
                         item['properties']['is_complete'] = True
                return processed, []
            tasks.append(_fetch_by_dois())

        # Tasks for searching by Titles
        if self.seed_paper_titles:
            async def _fetch_by_title(title):
                logging.info(f"Fetching papers by title: '{title[:50]}...'")
                # Use the wrapper for the S2 call
                s2_paper_metadata = await self._s2_api_call_wrapper(
                    self.s2.search_paper_by_keywords(query=title, fields_of_study=fields_of_study, limit=limit)
                )
                seed_meta, searched_meta = [], []
                if s2_paper_metadata:
                    seed_meta.append(s2_paper_metadata[0])
                    searched_meta.extend(s2_paper_metadata[1:])
                logging.info(f"Processing {len(seed_meta)} seed and {len(searched_meta)} searched papers for title '{title[:50]}...'")
                processed_seed = process_paper_metadata(seed_meta, from_dt, to_dt, fields_of_study)
                processed_searched = process_paper_metadata(searched_meta, from_dt, to_dt, fields_of_study)
                for item in processed_seed:
                     if item.get('type') == 'node' and item.get('labels') == ['Paper']:
                         item.setdefault('properties', {})['from_seed'] = True
                         item['properties']['from_title_search'] = True
                         item['properties']['is_complete'] = True
                for item in processed_searched:
                     if item.get('type') == 'node' and item.get('labels') == ['Paper']:
                         item.setdefault('properties', {})['from_title_search'] = True
                         item['properties']['is_complete'] = True
                return processed_seed, processed_searched

            for title in self.seed_paper_titles:
                 tasks.append(_fetch_by_title(title))


        # Task for searching by Research Topic
        if self.research_topic:
            async def _fetch_by_topic():
                logging.info(f"Fetching papers by topic: '{self.research_topic[:50]}...'")
                # Use the wrapper for the S2 call
                s2_paper_metadata = await self._s2_api_call_wrapper(
                    self.s2.search_paper_by_keywords(query=self.research_topic, fields=fields_of_study, limit=limit)
                )
                logging.info(f"Processing {len(s2_paper_metadata)} papers from topic search.")
                processed = process_paper_metadata(s2_paper_metadata, from_dt, to_dt, fields_of_study)
                for item in processed:
                     if item.get('type') == 'node' and item.get('labels') == ['Paper']:
                         item.setdefault('properties', {})['from_topic_search'] = True
                         item['properties']['is_complete'] = True
                return [], processed
            tasks.append(_fetch_by_topic())

        # Run all initial query tasks concurrently
        all_results = []
        if tasks:
            logging.info(f"Running {len(tasks)} initial query tasks concurrently...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            all_results.extend(results)
        else:
            logging.warning("No initial query criteria (DOI, Title, Topic) provided.")
            return

        # Process results and add to graph state
        logging.info("Aggregating results from initial queries...")
        aggregated_items = []
        for result in all_results:
            if isinstance(result, Exception):
                logging.error(f"Initial query task failed: {result}")
            elif isinstance(result, tuple) and len(result) == 2:
                seed_items, searched_items = result
                aggregated_items.extend(seed_items)
                aggregated_items.extend(searched_items)
            else:
                logging.warning(f"Unexpected result type from initial query task: {type(result)}")

        self._add_items_to_graph(aggregated_items)
        logging.info(f"Initial query complete. Nodes: {len(self.nodes_json)}, Edges: {len(self.edges_json)}")


    async def get_author_info(
            self,
            author_ids: List[str],
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Fetches and processes author information asynchronously."""
        if not author_ids: return []
        fields_of_study = fields_of_study if fields_of_study is not None else self.fields_of_study
        from_dt = from_dt if from_dt is not None else self.from_dt
        to_dt = to_dt if to_dt is not None else self.to_dt

        logging.info(f"Fetching info for {len(author_ids)} authors...")
        # Use the wrapper for the S2 call
        authors_info = await self._s2_api_call_wrapper(
            self.s2.search_author_by_ids(author_ids=author_ids, with_abstract=True )
        )

        logging.info(f"Processing metadata for {len(authors_info)} authors.")
        s2_author_meta_json = process_author_metadata(
            s2_author_metadata=authors_info,
            from_dt=from_dt,
            to_dt=to_dt,
            fields_of_study=fields_of_study 
        )

        for item in s2_author_meta_json:
            if item.get('type') == 'node':
                item.setdefault('properties', {})
                if item.get('labels') == ['Paper']:
                    item['properties']['from_same_author'] = True
                    item['properties']['is_complete'] = True
                elif item.get('labels') == ['Author']:
                    item['properties']['is_complete'] = True
        return s2_author_meta_json


    async def get_cited_papers( # References
            self,
            paper_doi: str,
            limit: Optional[int] = 100,
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Get papers cited by the paper (references) asynchronously."""
        fields_of_study = fields_of_study if fields_of_study is not None else self.fields_of_study
        from_dt = from_dt if from_dt is not None else self.from_dt
        to_dt = to_dt if to_dt is not None else self.to_dt

        logging.info(f"Fetching papers cited by {paper_doi}...")
         # Use the wrapper for the S2 call
        s2_citedpaper_metadata_raw = await self._s2_api_call_wrapper(
            self.s2.get_s2_cited_papers( # This calls get_paper_references in kit
                paper_id=paper_doi,
                limit=limit,
                with_abstract=True # Let the kit handle fetching missing abstracts
                )
        )

        logging.info(f"Processing {len(s2_citedpaper_metadata_raw)} cited papers (references) for {paper_doi}.")
        s2_citedpapermeta_json = process_citation_metadata(
            original_paper_doi=paper_doi,
            s2_citation_metadata=s2_citedpaper_metadata_raw, # Pass the raw result from S2 kit
            citation_type='citedPaper', # Reference (paper cites this)
            from_dt=from_dt,
            to_dt=to_dt,
            fields_of_study=fields_of_study 
        )

        for item in s2_citedpapermeta_json:
            if item.get('type') == 'node' and item.get('labels') == ['Paper']:
                item.setdefault('properties', {})['from_reference'] = True
                item['properties']['is_complete'] = True
        return s2_citedpapermeta_json


    async def get_citing_papers( # Citations
            self,
            paper_doi: str,
            limit: Optional[int] = 100,
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Retrieve papers that cite the paper asynchronously."""
        fields_of_study = fields_of_study if fields_of_study is not None else self.fields_of_study
        from_dt = from_dt if from_dt is not None else self.from_dt
        to_dt = to_dt if to_dt is not None else self.to_dt

        logging.info(f"Fetching papers citing {paper_doi}...")
        # Use the wrapper for the S2 call
        s2_citingpaper_metadata_raw = await self._s2_api_call_wrapper(
            self.s2.get_s2_citing_papers( # This calls get_paper_citations in kit
                paper_id=paper_doi,
                limit=limit,
                with_abstract=True # Let the kit handle fetching missing abstracts
                )
        )

        logging.info(f"Processing {len(s2_citingpaper_metadata_raw)} citing papers for {paper_doi}.")
        s2_citingpapermetadata_json = process_citation_metadata(
            original_paper_doi=paper_doi,
            s2_citation_metadata=s2_citingpaper_metadata_raw, # Pass the raw result
            citation_type='citingPaper', # Citation (this paper cites original)
            from_dt=from_dt,
            to_dt=to_dt,
            fields_of_study=fields_of_study # Pass fields for filtering
        )

        for item in s2_citingpapermetadata_json:
            if item.get('type') == 'node' and item.get('labels') == ['Paper']:
                item.setdefault('properties', {})['from_citation'] = True
                item['properties']['is_complete'] = True
        return s2_citingpapermetadata_json

    async def get_recommend_papers(
            self,
            paper_dois: List[str],
            limit: Optional[int] = 100,
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Retrieve recommended papers asynchronously."""
        if not paper_dois: return []
        fields_of_study = fields_of_study if fields_of_study is not None else self.fields_of_study
        from_dt = from_dt if from_dt is not None else self.from_dt
        to_dt = to_dt if to_dt is not None else self.to_dt

        logging.info(f"Fetching recommendations based on {len(paper_dois)} papers...")
         # Use the wrapper for the S2 call
        s2_recommended_metadata = await self._s2_api_call_wrapper(
             self.s2.get_s2_recommended_papers(
                positive_paper_ids=paper_dois,
                limit=limit,
                with_abstract=True # Let kit handle abstract fetching
                )
        )

        logging.info(f"Processing {len(s2_recommended_metadata)} recommended papers.")
        s2_recpapermetadata_json = process_paper_metadata( # Use standard paper processing
            s2_paper_metadata=s2_recommended_metadata,
            from_dt=from_dt,
            to_dt=to_dt,
            fields_of_study=fields_of_study
        )

        for item in s2_recpapermetadata_json:
            if item.get('type') == 'node' and item.get('labels') == ['Paper']:
                item.setdefault('properties', {})['from_recommended'] = True
                item['properties']['is_complete'] = True
        return s2_recpapermetadata_json


    async def llm_gen_related_topics(self, seed_paper_ids: List[str]) -> Optional[Dict]:
        """Generate related topics using LLM asynchronously."""
        logging.info(f"Generating related topics for {len(seed_paper_ids)} seed papers...")
        # Use existing node data
        nodes_dict = {node['id']: node for node in self.nodes_json}
        domains, seed_paper_texts = [], []

        for paper_id in seed_paper_ids:
            item = nodes_dict.get(paper_id)
            if item and item.get('labels') == ['Paper']:
                props = item.get('properties', {})
                title = props.get('title')
                abstract = props.get('abstract')
                # Handle both list and potentially string fieldsOfStudy
                domain_info = props.get('fieldsOfStudy')
                paper_domains = []
                if isinstance(domain_info, list):
                     paper_domains = domain_info
                elif isinstance(domain_info, str): # Handle case if it's just a string
                     paper_domains = [domain_info]

                if title and abstract:
                    info = f"<paper>\nTitle: {title}\nAbstract: {abstract}\n</paper>"
                    seed_paper_texts.append(info)
                if paper_domains:
                    domains.extend(paper_domains)
            else:
                 logging.warning(f"Seed paper ID {paper_id} not found or not a Paper node.")

        if not seed_paper_texts:
            logging.error("No text found for seed papers to generate topics.")
            return None
        if not domains:
            logging.warning("No domains found for seed papers, using 'General Science'.")
            domain = "General Science"
        else:
            domain_counts = Counter(d for d in domains if d) # Count non-empty domains
            domain = domain_counts.most_common(1)[0][0] if domain_counts else "General Science"


        # LLM call (using async retry version)
        qa_prompt = keywords_topics_prompt.format(
            domain=domain,
            example_json=keywords_topics_example,
            input_text="\n\n".join(seed_paper_texts)
        )
        logging.info("Calling LLM to generate topics...")
        keywords_topics_json = None
        try:
            # Use the ASYNC retry wrapper
            keywords_topics_info = await async_llm_gen_w_retry(
                 self.llm_api_key, self.llm_model_name, qa_prompt, sys_prompt=None, temperature=0.6
                 )

            if keywords_topics_info:
                repaired_json_str = repair_json(keywords_topics_info)
                keywords_topics_json = json.loads(repaired_json_str)
                logging.info(f"LLM generated topics: {json.dumps(keywords_topics_json)}")
            else:
                 logging.error("LLM returned empty or None response for topic generation after retries.")
                 return None

        except json.JSONDecodeError as e:
            logging.error(f"LLM Topic Generation - JSON Repair/Decode Error: {e}. Original output: {keywords_topics_info if 'keywords_topics_info' in locals() else 'N/A'}")
            return None
        except Exception as e:
            logging.error(f"Error during LLM topic generation or processing: {e}")
            return None

        # Add Topic nodes and relationships (synchronous part, safe after await)
        processed_topic_items = []
        query_topics = keywords_topics_json.get('queries', [])
        current_node_ids = self._node_ids # Use the class attribute set

        for topic in query_topics:
            if not isinstance(topic, str) or not topic: continue # Skip empty/invalid topics
            topic_hash_id = generate_hash_key(topic)
            if topic_hash_id not in current_node_ids:
                topic_node = {
                    'type': 'node', 'id': topic_hash_id, 'labels': ['Topic'],
                    'properties': {'topic_hash_id': topic_hash_id, 'topic_name': topic, 'hash_method': 'hashlib.sha256'}
                }
                processed_topic_items.append(topic_node)
                # Don't add to self._node_ids here, wait for _add_items_to_graph

            for paper_id in seed_paper_ids:
                 if paper_id in nodes_dict:
                    edge_tuple_key = (paper_id, topic_hash_id, "DISCUSS")
                    if edge_tuple_key not in self._edge_tuples:
                        paper_topic_relationship = {
                            "type": "relationship", "relationshipType": "DISCUSS",
                            "startNodeId": paper_id, "endNodeId": topic_hash_id,
                            "properties": {}
                        }
                        processed_topic_items.append(paper_topic_relationship)
                        # Don't add to self._edge_tuples here

        # Add newly created topic nodes/edges to the main graph state centrally
        self._add_items_to_graph(processed_topic_items)
        logging.info(f"Added items related to {len(query_topics)} generated topics.")

        return keywords_topics_json


    async def get_related_papers(
            self,
            topic_json: Dict,
            limit: Optional[int] = 100,
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Fetch related papers based on LLM queries concurrently."""
        queries = topic_json.get('queries', [])
        if not queries or not isinstance(queries, list):
            logging.warning("No valid queries found in topic_json for related paper search.")
            return []

        fields_of_study = fields_of_study if fields_of_study is not None else self.fields_of_study
        from_dt = from_dt if from_dt is not None else self.from_dt
        to_dt = to_dt if to_dt is not None else self.to_dt

        tasks = []
        async def _fetch_related_for_query(topic):
            if not isinstance(topic, str) or not topic: return [] # Skip invalid queries
            logging.info(f"Fetching related papers for query: '{topic[:50]}...'")
             # Use the wrapper for the S2 call
            s2_paper_metadata = await self._s2_api_call_wrapper(
                self.s2.search_paper_by_keywords(
                    query=topic,
                    fields_of_study=fields_of_study, # Pass fields
                    limit=limit)
            )
            logging.info(f"Processing {len(s2_paper_metadata)} related papers for query '{topic[:50]}...'")
            # Use standard paper processing
            s2_papermeta_json = process_related_metadata(
                s2_paper_metadata=s2_paper_metadata,
                topic=topic,
                from_dt=from_dt,
                to_dt=to_dt,
                fields_of_study=fields_of_study)

            for item in s2_papermeta_json:
                if item.get('type') == 'node' and item.get('labels') == ['Paper']:
                    item.setdefault('properties', {})['from_related_topics'] = True
                    item['properties']['is_complete'] = True
            return s2_papermeta_json

        for topic in queries:
            tasks.append(_fetch_related_for_query(topic))

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
        # (Code remains largely the same as previous async version)
        semantic_similar_pool = []
        publish_dt_ref = {
            x['id']: x['properties'].get('publicationDate')
            for x in paper_nodes_json if x.get('id') and x.get('properties') and x['properties'].get('publicationDate')
        }

        ids, texts = [], []
        for node in paper_nodes_json:
             if not isinstance(node, dict): continue
             node_id = node.get('id')
             props = node.get('properties', {})
             title = props.get('title')
             abstract = props.get('abstract')
             if node_id and title and abstract:
                texts.append(f"Title: {title}\nAbstract: {abstract}") # Simpler format
                ids.append(node_id)

        if not texts:
            logging.warning("No paper titles and abstracts found for semantic similarity calculation.")
            return []

        logging.info(f"Generating embeddings for {len(texts)} papers...")
        try:
             embeds = await gemini_embedding_async(self.embed_api_key, self.embed_model_name, texts, 10)
        except Exception as e:
             logging.error(f"Embedding generation failed: {e}")
             return []

        if not embeds or len(embeds) != len(texts):
             logging.error(f"Embedding generation returned unexpected result. Expected {len(texts)}, got {len(embeds) if embeds else 0}.")
             return []

        logging.info("Calculating similarity matrix...")
        try:
            # If semantic_similarity_matrix is very CPU intensive and blocks, wrap it:
            loop = asyncio.get_running_loop()
            sim_matrix = await loop.run_in_executor(None, semantic_similarity_matrix, embeds, embeds)
            # Otherwise, call directly if it's fast enough:
            # sim_matrix = semantic_similarity_matrix(embeds, embeds)
            sim_matrix = np.array(sim_matrix) # Ensure numpy array
        except Exception as e:
            logging.error(f"Similarity matrix calculation failed: {e}")
            return []

        logging.info("Processing similarity matrix to create relationships...")
        rows, cols = sim_matrix.shape
        added_pairs = set() # To store tuples of (id1, id2) where id1 < id2

        for i in range(rows):
            for j in range(i + 1, cols): # Iterate upper triangle only (j > i)
                sim = sim_matrix[i, j]
                id_i = ids[i]
                id_j = ids[j]

                # Determine start/end node consistently (lexicographically)
                start_node_id, end_node_id = min(id_i, id_j), max(id_i, id_j)

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
                    added_pairs.add(pair_tuple)

        logging.info(f"Generated {len(semantic_similar_pool)} potential similarity relationships.")
        return semantic_similar_pool


    async def init_collect(
            self,
            limit=50,
            from_dt='2022-01-01',
            to_dt='2025-04-01', # Use current date or future if appropriate
            fields_of_study=None,
    ):
        """Initializes the collection by running the initial paper query."""
        await self.initial_paper_query(limit=limit, from_dt=from_dt, to_dt=to_dt, fields_of_study=fields_of_study)


    # The main optimized async collect method
    async def collect(
            self,
            seed_paper_dois: List[str],
            with_reference: Optional[bool] = True,
            with_author: Optional[bool] = True,
            with_recommend: Optional[bool] = True,
            with_expanded_search: Optional[bool] = True,
            add_semantic_similarity: Optional[bool] = True,
            similarity_threshold: Optional[float] = 0.7,
            limit=50,
            from_dt='2022-01-01',
            to_dt='2025-04-01', # Use current date or future if appropriate
            fields_of_study=None,
    ):
        """
        Main asynchronous collection method orchestrating various data fetching tasks.
        """
        if not seed_paper_dois:
             logging.warning("Collect called with no seed paper DOIs.")
             # Optionally exit or proceed based on requirements
             # return

        fields_of_study = fields_of_study if fields_of_study is not None else self.fields_of_study

        data_fetching_tasks = []
        processed_items_aggregate = []

        # --- Task Prep: Author Info ---
        if with_author:
            author_ids_to_fetch = set()
            current_nodes_dict = {node['id']: node for node in self.nodes_json}
            current_complete_author_ids = {node['id'] for node in self.nodes_json if node.get('labels') == ["Author"] and node.get('properties', {}).get('is_complete')}

            for doi in seed_paper_dois:
                 paper_node = current_nodes_dict.get(doi)
                 if paper_node and paper_node.get('labels') == ["Paper"]:
                     authors = paper_node.get('properties', {}).get('authors', [])
                     for author in authors:
                         author_id = author.get('authorId')
                         if author_id and author_id not in current_complete_author_ids:
                             author_ids_to_fetch.add(author_id)

            if author_ids_to_fetch:
                 logging.info(f"Preparing author info task for {len(author_ids_to_fetch)} authors.")
                 data_fetching_tasks.append(self.get_author_info(list(author_ids_to_fetch), from_dt, to_dt, fields_of_study))
            else:
                 logging.info("No new authors found for seed DOIs or all are already complete.")

        # --- Task Prep: References & Citations ---
        if with_reference and seed_paper_dois:
            logging.info(f"Preparing reference/citation tasks for {len(seed_paper_dois)} seed papers.")
            for paper_doi in seed_paper_dois:
                # Pass combined fields list
                data_fetching_tasks.append(self.get_cited_papers(paper_doi, limit, from_dt, to_dt, fields_of_study))
                data_fetching_tasks.append(self.get_citing_papers(paper_doi, limit, from_dt, to_dt, fields_of_study))

        # --- Task Prep: Recommendations ---
        if with_recommend and seed_paper_dois:
             logging.info("Preparing recommendations task.")
             data_fetching_tasks.append(self.get_recommend_papers(seed_paper_dois, limit, from_dt, to_dt, fields_of_study))

        # --- Task Prep: Expanded Search ---
        expanded_search_task = None
        if with_expanded_search and seed_paper_dois:
            async def _handle_expanded_search():
                logging.info("Starting expanded search sub-task...")
                # LLM gen adds topic nodes/edges directly via _add_items_to_graph
                keywords_topics_json = await self.llm_gen_related_topics(seed_paper_dois)
                related_papers_items = []
                if keywords_topics_json:
                    # Fetch related papers based on topics
                    related_papers_items = await self.get_related_papers(
                        keywords_topics_json, limit, from_dt, to_dt, fields_of_study
                        )
                else:
                    logging.warning("Skipping related paper fetch as no topics were generated.")
                return related_papers_items # Return newly fetched related papers

            logging.info("Preparing expanded search task.")
            expanded_search_task = asyncio.create_task(_handle_expanded_search())

        # --- Execute Concurrent Data Fetching Tasks ---
        if data_fetching_tasks:
            logging.info(f"Running {len(data_fetching_tasks)} main data collection tasks concurrently...")
            results = await asyncio.gather(*data_fetching_tasks, return_exceptions=True)
            logging.info("Main data collection tasks finished.")
            for result in results:
                if isinstance(result, Exception):
                    logging.error(f"Data collection task failed: {result}")
                elif isinstance(result, list):
                    processed_items_aggregate.extend(result) # Collect items
        else:
            logging.info("No main data collection tasks to run.")

        # --- Wait for and Process Expanded Search Task Results ---
        if expanded_search_task:
            logging.info("Waiting for expanded search task to complete...")
            try:
                expanded_result = await expanded_search_task
                if isinstance(expanded_result, list):
                     processed_items_aggregate.extend(expanded_result) # Add related papers
                logging.info("Expanded search task finished.")
            except Exception as e:
                 logging.error(f"Expanded search task failed: {e}")

        # --- Add all aggregated items to the graph state ---
        logging.info(f"Adding {len(processed_items_aggregate)} items from tasks to graph...")
        self._add_items_to_graph(processed_items_aggregate)
        logging.info(f"Graph state after collection tasks. Nodes: {len(self.nodes_json)}, Edges: {len(self.edges_json)}")

        # --- Task: Semantic Similarity ---
        if add_semantic_similarity:
            logging.info("Starting semantic similarity calculation...")
            # Use the latest graph state
            paper_nodes_for_similarity = [node for node in self.nodes_json if node.get('labels') == ["Paper"]]
            if paper_nodes_for_similarity:
                try:
                    semantic_similar_pool = await self.cal_semantic_similarity(paper_nodes_for_similarity)
                    semantic_similar_relationships = [
                        edge for edge in semantic_similar_pool
                        if edge.get('properties', {}).get('weight', 0) > similarity_threshold
                    ]
                    logging.info(f"Adding {len(semantic_similar_relationships)} similarity edges above threshold {similarity_threshold}...")
                    self._add_items_to_graph(semantic_similar_relationships)
                except Exception as e:
                    logging.error(f"Semantic similarity calculation or addition failed: {e}")
            else:
                 logging.warning("No paper nodes found to calculate semantic similarity.")

        # Final state log
        logging.info(f"Collection process complete. Final state: Nodes={len(self.nodes_json)}, Edges={len(self.edges_json)}")

import os
llm_api_key = os.getenv('GEMINI_API_KEY_3')
llm_model_name="gemini-2.0-flash"
embed_api_key = os.getenv('GEMINI_API_KEY_3')
embed_model_name="models/text-embedding-004"

# --- Example Usage (similar to before, adjust parameters if needed) ---
async def main():
    searcher = PaperSearch(
        seed_paper_dois=["10.1109/CVPR.2016.90"], # Example DOI
        llm_api_key=llm_api_key,
        llm_model_name=llm_model_name,
        embed_api_key=embed_api_key,
        embed_model_name=embed_model_name,
        # --- Control Concurrency/Rate ---
        s2_max_concurrent=1, # Allow up to 5 S2 requests at once
        s2_request_interval=3.5 # Target avg 3.5s between S2 requests (~0.28/sec)
    )


    print("--- Running Initial Query ---")
    fields_of_study = ['Computer Science']
    await searcher.init_collect(limit=10, from_dt="2015-01-01", to_dt="2025-04-01", fields_of_study=fields_of_study)
    print(f"Nodes after init: {len(searcher.nodes_json)}")
    print(f"Edges after init: {len(searcher.edges_json)}")

    seed_dois_in_graph = [node['id'] for node in searcher.nodes_json if node.get('labels') == ['Paper'] and node.get('properties', {}).get('from_seed')]
    if not seed_dois_in_graph:
         print("Warning: Seed DOI(s) not found after initial query.")
         return

    print(f"\n--- Running Main Collection for seed DOIs: {seed_dois_in_graph} ---")
    start_time = asyncio.get_event_loop().time()

    await searcher.collect(
        seed_paper_dois=seed_dois_in_graph,
        with_reference=True,
        with_author=True,
        with_recommend=True,
        with_expanded_search=False, # Set True to test LLM part
        add_semantic_similarity=True,
        similarity_threshold=0.75,
        limit=20,
        from_dt="2015-01-01",
        to_dt="2025-04-01",
        fields_of_study=fields_of_study # Pass desired fields for collect phase
    )
    end_time = asyncio.get_event_loop().time()
    print(f"\nOptimized collection took {end_time - start_time:.2f} seconds.")
    print(f"Final Nodes: {len(searcher.nodes_json)}")
    print(f"Final Edges: {len(searcher.edges_json)}")

if __name__ == "__main__":
    asyncio.run(main())