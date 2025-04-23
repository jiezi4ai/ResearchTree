import json
import asyncio
from collections import Counter
from json_repair import repair_json
from typing import List, Dict, Optional, Union, Tuple

from apis.s2_api import SemanticScholarKit

from utils.data_process import generate_hash_key
from models.llms import async_llm_gen_w_retry
from apis.s2_data_process import process_related_metadata

from prompts.query_prompt import keywords_topics_example, keywords_topics_prompt


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TopicQuery:
    def __init__(
            self,
            nodes_json: Optional[List[Dict]] = None,
            edges_json: Optional[List[Dict]] = None, 
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
    ):
        # initiate semantic scholar
        self.s2 = SemanticScholarKit() 

        # State: Nodes and Edges
        self.nodes_json = nodes_json if nodes_json else []
        self.edges_json = edges_json if edges_json else []

        # Use a set for faster node/edge existence checks during addition
        self._node_ids = set()
        self._edge_tuples = set() # Store (start_id, end_id, type) tuples

        # Filters
        self.from_dt = from_dt
        self.to_dt = to_dt
        self.fields_of_study = fields_of_study


    async def llm_gen_related_topics(
            self, 
            seed_paper_json: List[Dict],
            llm_api_key: str,
            llm_model_name: str,
            ) -> Optional[Dict]:
        """Generate related topics using LLM asynchronously."""
        logging.info(f"Generating related topics for {len(seed_paper_json)} seed papers...")
        # Use existing node data, assuming initial query ran
        nodes_dict = {node['id']: node for node in seed_paper_json}
        seed_paper_ids = [node['id'] for node in seed_paper_json]
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
            return {}
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
            keywords_topics_info = await async_llm_gen_w_retry(llm_api_key, llm_model_name, qa_prompt, sys_prompt=None, temperature=0.6)
            if not keywords_topics_info:
                 logging.error("LLM returned empty response for topic generation.")
                 return {}

            # Repair and parse JSON
            repaired_json_str = repair_json(keywords_topics_info)
            keywords_topics_json = json.loads(repaired_json_str)
            logging.info(f"LLM generated topics: {json.dumps(keywords_topics_json)}")

        except json.JSONDecodeError as e:
            logging.error(f"LLM Topic Generation - JSON Repair/Decode Error: {e}. Original output: {keywords_topics_info}")
            return {}
        except Exception as e:
            logging.error(f"Error during LLM topic generation or processing: {e}")
            return {}

        return keywords_topics_json
    

    def get_topic_json(self, keywords_topics_json, seed_paper_json):
        """Add Topic nodes and relationships"""
        processed_json = []
        existing_node_ids = set([x['id'] for x in processed_json if x['type']=='node'])
        existing_edge_ids = set([(x['startNodeId'],x['endNodeId']) 
                                 for x in processed_json if x['type']=='relationship'])

        seed_paper_dois = {node['id'] for node in seed_paper_json} # Get current node IDs
        query_topics = keywords_topics_json.get('queries', []) # Default to empty list

        for topic in query_topics:
            topic_hash_id = generate_hash_key(topic)
            
            if topic_hash_id not in existing_node_ids:
                topic_node = {
                    'type': 'node',
                    'id': topic_hash_id,
                    'labels': ['Topic'],
                    'properties': {'topic_hash_id': topic_hash_id, 'topic_name': topic, 'hash_method': 'hashlib.sha256'}
                }
                processed_json.append(topic_node)
                existing_node_ids.add(topic_hash_id)

            for paper_id in seed_paper_dois:
                # Check if edge already exists using the internal set for efficiency
                edge_tuple = (paper_id, topic_hash_id, "DISCUSS")
                if edge_tuple not in existing_edge_ids:
                    paper_topic_relationship = {
                        "type": "relationship",
                        "relationshipType": "DISCUSS",
                        "startNodeId": paper_id,
                        "endNodeId": topic_hash_id,
                        "properties": {}
                    }
                    processed_json.append(paper_topic_relationship)
                    existing_edge_ids.add(edge_tuple) # Add locally

        return processed_json
    

    async def get_topic_papers(
            self,
            topic_queries: List[str],
            limit: Optional[int] = 100,
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
    ) -> List[Dict]: # Return aggregated processed items
        """Fetch related papers based on LLM queries concurrently."""
        # queries = topic_json.get('queries', [])
        if not isinstance(topic_queries, list) or len(topic_queries) == 0:
            logging.warning("No queries found from input.")
            return []

        fields_of_study = fields_of_study if fields_of_study is not None else self.fields_of_study
        from_dt = from_dt if from_dt is not None else self.from_dt
        to_dt = to_dt if to_dt is not None else self.to_dt

        tasks = []

        for query in topic_queries:
            logging.info(f"Fetching related papers for query: '{query[:50]}...'")
            coro = await self.s2.async_search_paper_by_keywords(query, fields_of_study=fields_of_study, limit=limit, match_title=False)
            tasks.append(coro)

        # Run all related paper searches concurrently
        related_topic_papers = []
        if tasks:
            logging.info(f"Running {len(tasks)} related paper search tasks concurrently...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logging.error(f"Related paper search task failed: {result}")
                elif isinstance(result, list):
                    # get paper json data
                    result_json = process_related_metadata( 
                        s2_related_metadata=result,
                        topic=query,
                        from_dt=from_dt,
                        to_dt=to_dt,
                        fields_of_study=fields_of_study
                    )
                    # Mark nodes
                    for item in result_json:
                        if item['type'] == 'node' and item['labels'] == ['Paper']:
                            item['properties']['from_related_topics'] = True
                            item['properties']['is_complete'] = True
                    related_topic_papers.extend(result_json)

        return related_topic_papers
