# paper_data_collect.py
# utilize semantic scholar to collect and process paper related metadata

# import external packages
import re
import json
import asyncio
from collections import Counter
from json_repair import repair_json
from typing import List, Dict, Optional, Union, Any, Set, Tuple

from semanticscholar.Paper import Paper
from semanticscholar.Author import Author
from semanticscholar.Citation import Citation
from semanticscholar.Reference import Reference
from semanticscholar.PaginatedResults import PaginatedResults

import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

# import internal packages
from apis.s2_api import SemanticScholarKit
from collect.paper_data_find import PaperFinder
from models.llms import async_llm_gen_w_retry
from collect.paper_data_process import (process_papers_data, process_authors_data, 
                                        process_citations_data, process_topics_data)
from prompts.query_prompt import keywords_topics_example, keywords_topics_prompt

import logging
# Configure logging
logger = logging.getLogger('SemanticScholarKit')
# Prevent duplicate handlers if the root logger is already configured
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
# Set default level (can be overridden by user)
logger.setLevel(logging.INFO) 


NODE_PAPER = "Paper"
NODE_AUTHOR = "Author"


async def llm_gen_topics(
        paper_json: List[Dict],
        llm_api_key: str,
        llm_model_name: str,
        ) -> Optional[Dict]:
    """Generate related topics using LLM asynchronously."""
    logger.info(f"Generating related topics for {len(paper_json)} seed papers...")
    # Use existing node data, assuming initial query ran
    nodes_dict = {node['id']: node for node in paper_json}
    seed_paper_ids = [node['id'] for node in paper_json]
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
                logger.warning(f"Seed paper ID {paper_id} not found or not a Paper node.")

    if not seed_paper_texts:
        logger.error("No text found for seed papers to generate topics.")
        return {}
    if not domains:
        logger.warning("No domains found for seed papers, using 'General Science'.")
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
    logger.info("Calling LLM to generate topics...")
    try:
        # Assuming llm_gen_w_retry is async
        keywords_topics_info = await async_llm_gen_w_retry(llm_api_key, llm_model_name, qa_prompt, sys_prompt=None, temperature=0.6)
        if not keywords_topics_info:
                logger.error("LLM returned empty response for topic generation.")
                return {}

        # Repair and parse JSON
        repaired_json_str = repair_json(keywords_topics_info)
        keywords_topics_json = json.loads(repaired_json_str)
        logger.info(f"LLM generated topics: {json.dumps(keywords_topics_json)}")

    except json.JSONDecodeError as e:
        logger.error(f"LLM Topic Generation - JSON Repair/Decode Error: {e}. Original output: {keywords_topics_info}")
        return {}
    except Exception as e:
        logger.error(f"Error during LLM topic generation or processing: {e}")
        return {}

    return keywords_topics_json


class PaperCollector(PaperFinder):
    """
    A class for exploring academic papers, optimized for asynchronous operations.
    """
    def __init__(
            self,
            # seed papers
            seed_research_topics: Optional[List] = None,
            seed_paper_titles: Optional[Union[List[str], str]] = None,
            seed_paper_ids: Optional[Union[List[str], str]] = None,
            # parameters
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
            nodes_json: Optional[List[Dict]] = None,
            edges_json: Optional[List[Dict]] = None, 
            author_paper_limit: Optional[int] = 10,
            search_limit: Optional[int] = 50,
            recommend_limit: Optional[int] = 50,
            citation_limit: Optional[int] = 100,
            # instances
            s2_instance: Optional[SemanticScholarKit] = None, 
    ):
        """Initialize PaperCollector parameters."""
        super().__init__(
            # seed papers
            seed_research_topics = seed_research_topics,
            seed_paper_titles = seed_paper_titles,
            seed_paper_ids = seed_paper_ids,
            # parameters
            from_dt = from_dt,
            to_dt = to_dt,
            fields_of_study = fields_of_study,
            author_paper_limit = author_paper_limit,
            search_limit = search_limit,
            recommend_limit = recommend_limit,
            citation_limit = citation_limit,
            # instances
            s2_instance = s2_instance 
        )

        # ------- GLOBAL STATUS VARIABLES ------
        # initiate semantic scholar instances
        self.s2 = s2_instance if s2_instance and isinstance(s2_instance, SemanticScholarKit) else SemanticScholarKit()

        # State: Nodes and Edges
        self.nodes_json: List[Dict] = list(nodes_json) if nodes_json else []
        self.edges_json: List[Dict] = list(edges_json) if edges_json else []


    ############################################################################
    # paper search relatied functions
    ############################################################################
    async def topic_generation(
            self,
            paper_json: List[Dict],
            llm_api_key: str,
            llm_model_name: str
            ):
        """use LLM to generate related topics based on seed paper information"""
        logging.info("Use LLM to identify key related topics.")
        keywords_topics_json = await llm_gen_topics(paper_json, llm_api_key, llm_model_name)
        query_topics = keywords_topics_json.get('queries', []) 
        paper_ids = [item['id'] for item in paper_json]

        if not query_topics or not paper_ids:
            return

        for topic in set(query_topics):
            for pid in set(paper_ids):
                self.data_pool['topic'].append({'topic': topic, 'paperId': pid})


    ############################################################################
    # paper consolidated search
    ############################################################################
    async def consolidated_search(
            self,
            # for paper / author info
            topics: Optional[List] = None,
            paper_titles: Optional[List] = None,
            paper_ids: Optional[List] = None,
            author_ids: Optional[List] = None,
            author_paper_ids: Optional[List[str]] = None, # Paper IDs to fetch authors for
            # for citation info, be very careful to use s2 paper ids
            ref_paper_ids: Optional[List] = None,
            cit_paper_ids: Optional[List] = None,
            # for S2 recommendations, be very careful to use s2 paper ids
            pos_paper_ids: Optional[List] = None,
            neg_paper_ids: Optional[List] = None,
            # search params
            author_limit: Optional[int] = None, # Limit for fetch_authors_for_papers
            search_limit: Optional[int] = None,
            citation_limit: Optional[int] = None,
            recommend_limit: Optional[int] = None,
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List] = None
        ):
        """
        Consolidates various search types into a single asynchronous execution.
        """
        tasks = []
        logger.info("consolidated_search: Starting...")

        # --- Paper Info ---
        if paper_titles or paper_ids:
            tasks.append(self.paper_search(paper_titles, paper_ids))
        if topics:
            tasks.append(self.topic_search(topics, search_limit, from_dt, to_dt, fields_of_study))  

        # --- Author Info ---
        if author_ids:
            tasks.append(self.authors_search(author_ids))
        if author_paper_ids:
            tasks.append(self.paper_author_search(author_paper_ids, author_limit))

        # --- Citation Info ---
        if ref_paper_ids:
            tasks.append(self.reference_search(ref_paper_ids, citation_limit))
        if cit_paper_ids:
            tasks.append(self.citing_search(cit_paper_ids, citation_limit))

        # --- Recommendations ---
        if pos_paper_ids:
            tasks.append(self.paper_recommendation(pos_paper_ids, neg_paper_ids, recommend_limit))

        # --- Execute All ---
        if tasks:
            logger.info(f"consolidated_search: Running {len(tasks)} sub-tasks concurrently...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("consolidated_search: Sub-tasks finished.")
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Identify the failed task (requires more complex tracking or assumptions)
                    logger.error(f"consolidated_search: A sub-task failed: {result}", exc_info=True) # Log traceback
        else:
            logger.warning("consolidated_search: No search criteria provided.")

        logger.info("consolidated_search: Finished.")


    ############################################################################
    # paper search results post processing
    ############################################################################
    async def supplement_abstract(self, paper_ids:Union[List[str], str]) -> Dict[str, Optional[str]]:
        """
        Fetches abstracts for given paper IDs from Semantic Scholar.
        Returns a dictionary mapping paperId to abstract.
        paper_ids: already filter out any IDs with abstracts
        """
        if not paper_ids:
            return {}

        logger.info(f"supplement_abstract: Fetching abstracts for {len(paper_ids)} papers...")
        papers_abstract_info: List[Paper] = []
        papers_abstract_info = await self.s2.get_papers(paper_ids=paper_ids, fields=['paperId', 'abstract'])

        papers_abstract_ref: Dict[str, Optional[str]] = {pid: None for pid in paper_ids} # Initialize
        if papers_abstract_info:
            for item in papers_abstract_info:
                 if isinstance(item, Paper) and item.paperId in papers_abstract_ref:
                     papers_abstract_ref[item.paperId] = item.abstract # Will be None if S2 doesn't have it

        found_count = sum(1 for v in papers_abstract_ref.values() if v is not None)
        logger.info(f"supplement_abstract: Found abstracts for {found_count}/{len(paper_ids)} papers.")
        
        return papers_abstract_ref


    async def post_process(self, if_supplement_abstract: Optional[bool] = True):
        """
        Processes the raw data collected in self.data_pool using s2_data_process functions.
        Populates self.nodes_json and self.edges_json with Neo4j-compatible dictionaries.
        Handles deduplication across different data types processed.
        Optionally supplements missing abstracts.
        """
        logger.info("post_process: Starting data processing...")
        nodes_edges_json = []

        # Initialize sets here to pass across processing functions
        _node_ids: Set[str] = set()
        _edge_tuples: Set[Tuple[str, str, str]] = set()

        # --- 1. Process Papers ---
        if self.data_pool['paper']:
            logger.info(f"Processing {len(self.data_pool['paper'])} raw paper entries...")
            papers_json = process_papers_data(
                self.data_pool['paper'],
                existing_nodes=_node_ids,
                existing_edges=_edge_tuples)
            logger.info(f"Generated {len(papers_json)} nodes/edges from papers.")

            # supplement abstract for papers
            if if_supplement_abstract:
                null_abstract_ids = []
                paper_nodes_indices = {} # Store index to update later
                for idx, item in enumerate(papers_json):
                    if (item.get('type') == 'node' and NODE_PAPER in item.get('labels') 
                        and item.get('properties', {}).get('abstract') is None):
                        paper_id = item['id']
                        null_abstract_ids.append(item['id'])
                        paper_nodes_indices[paper_id] = idx

                if null_abstract_ids:
                    logger.info(f"Found {len(null_abstract_ids)} paper nodes missing abstracts. Attempting to supplement...")
                    papers_abstract_info = await self.supplement_abstract(null_abstract_ids)
                    update_count = 0
                    for pid, abstract in papers_abstract_info.items():
                        if abstract is not None and pid in paper_nodes_indices:
                            node_index = paper_nodes_indices[pid]
                            # Ensure the item is still a node and has properties
                            if papers_json[node_index].get('type') == 'node' and 'properties' in papers_json[node_index]:
                                papers_json[node_index]['properties']['abstract'] = abstract
                                update_count += 1
                            else:
                                logger.warning(f"post_process: Could not update abstract for paper {pid} at index {node_index} - item structure changed unexpectedly.")
                    logger.info(f"Successfully supplemented abstracts for {update_count} papers.")
            
            nodes_edges_json.extend(papers_json)
            logger.info(f"Total items after paper processing: {len(nodes_edges_json)}")
        else:
            logger.info("No paper data in pool to process.")

        # --- 2. Process Authors ---
        if len(self.data_pool['author']) > 0:
            logger.info(f"Processing {len(self.data_pool['author'])} raw author entries...")
            authors_json = process_authors_data(
                self.data_pool['author'],
                existing_nodes=_node_ids,
                existing_edges=_edge_tuples)
            nodes_edges_json.extend(authors_json)
            logger.info(f"Generated {len(authors_json)} nodes/edges from authors. Total items: {len(nodes_edges_json)}")
        else:
            logger.info("No author data in pool to process.")
        
        # --- 3. Process Topics ---
        if len(self.data_pool['topic']) > 0:
            logger.info(f"Processing {len(self.data_pool['topic'])} raw topic link entries...")
            topics_json = process_topics_data(
                self.data_pool['topic'],
                existing_nodes=_node_ids,
                existing_edges=_edge_tuples)
            nodes_edges_json.extend(topics_json)
            logger.info(f"Generated {len(topics_json)} nodes/edges from topics. Total items: {len(nodes_edges_json)}")
        else:
            logger.info("No topic data in pool to process.")

        # --- 4. Process Citations (References & Citings combined) ---
        # Combine both reference and citing data as they produce the same CITES relationship type
        all_citations_data = self.data_pool['reference'] + self.data_pool['citing']
        if all_citations_data:
            logger.info(f"Processing {len(all_citations_data)} raw citation entries ({len(self.data_pool['reference'])} refs, {len(self.data_pool['citing'])} citings)...")
            # Pass the *updated* edge set (node set less relevant here)
            citations_json = process_citations_data(
                all_citations_data,
                existing_edges=_edge_tuples
            )
            nodes_edges_json.extend(citations_json)
            logger.info(f"Generated {len(citations_json)} citation relationships. Total items: {len(nodes_edges_json)}")
        else:
            logger.info("No citation data (references or citings) in pool to process.")

        # --- 5. Finalize Output ---
        # Separate nodes and edges into final lists
        for item_json in nodes_edges_json:
            if item_json.get('type') == 'node':
                self.nodes_json.append(item_json)
            elif item_json.get('type') == 'relationship':
                self.edges_json.append(item_json)
