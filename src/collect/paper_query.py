import asyncio
from typing import List, Dict, Optional, Union, Tuple

from apis.s2_api import SemanticScholarKit
from apis.s2_data_process import reset_and_filter_paper, process_paper_metadata

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PaperQuery:
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


    def post_process(self, source, s2_paper_metadata, from_dt, to_dt, fields_of_study):
        """Post process paper metadata"""
        if isinstance(s2_paper_metadata, list) and len(s2_paper_metadata) > 0:
            # reset id and filter papers 
            filtered_s2_paper_metadata = reset_and_filter_paper(
                        s2_paper_metadata=s2_paper_metadata,
                        from_dt=from_dt, 
                        to_dt=to_dt, 
                        fields_of_study=fields_of_study)
            
            # convert paper to json format (compatible with graph)
            processed = process_paper_metadata(filtered_s2_paper_metadata)

            # mark paper information based on source
            if source == 'doi':
                for item in processed:
                    if item['type'] == 'node' and item['labels'] == ['Paper']:
                        item['properties']['from_seed'] = True
                        item['properties']['is_complete'] = True

            if source == 'title':
                # Mark appropriately
                i = 0
                for item in processed:
                    if item['type'] == 'node' and item['labels'] == ['Paper']:
                        if i == 0:  # when searching from s2 using paper title, treat the first result as actual paper
                            item['properties']['from_seed'] = True
                            item['properties']['is_complete'] = True
                        else:
                            item['properties']['from_title_search'] = True
                            item['properties']['is_complete'] = True
                        i += 1
                    
            if source == 'topic':
                for item in processed:
                    if item['type'] == 'node' and item['labels'] == ['Paper']:
                        item['properties']['from_topic_search'] = True
                        item['properties']['is_complete'] = True
            
            return processed


    async def get_paper_info(
            self,
            research_topic: Optional[str] = None,
            seed_paper_titles: Optional[Union[List[str], str]] = None,
            seed_paper_dois: Optional[Union[List[str], str]] = None,
            limit: Optional[int] = 100,
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
    ):
        """
        Retrieve papers based on user's input text asynchronously.
        (Args documentation omitted for brevity)
        """
        tasks_with_source = []
        task_coroutines = []
        
        # Task for searching by DOIs
        if seed_paper_dois:
            logging.info(f"Fetching papers by {len(seed_paper_dois)} DOIs...")
            coro = self.s2.async_search_paper_by_ids(id_list=seed_paper_dois)
            tasks_with_source.append({'source':'doi', 'value': seed_paper_dois, 'result': coro})
            task_coroutines.append(coro)
        
        if seed_paper_titles:
            for title in seed_paper_titles:
                logging.info(f"Fetching papers by title: '{title}...'")
                coro = self.s2.async_search_paper_by_keywords(query=title, fields_of_study=fields_of_study, limit=limit)
                tasks_with_source.append({'source':'title', 'value':title, 'result': coro})
                task_coroutines.append(coro)
        
        if research_topic:
            logging.info(f"Fetching papers by topic: '{research_topic}...'")
            coro = self.s2.async_search_paper_by_keywords(query=research_topic, fields_of_study=fields_of_study, limit=limit)
            tasks_with_source.append({'source':'topic', 'value':research_topic, 'result': coro})
            task_coroutines.append(coro)

        # Run all initial query tasks concurrently
        all_results = []
        if task_coroutines:
            logging.info(f"Running {len(task_coroutines)} initial query tasks concurrently...")
            results = await asyncio.gather(*task_coroutines, return_exceptions=True)
            all_results.extend(results)

        # post-process output
        processed_results = []
        if len(all_results) > 0:
            for i, task_info in enumerate(tasks_with_source):
                source = task_info.get('source')
                value = task_info.get('value')
                result = all_results[i]
                if isinstance(result, Exception):
                    logging.error(f"Task for source '{source}' ({value}) failed: {result}", exc_info=result)
                elif result is not None:
                    processed_result = self.post_process(source, result, from_dt, to_dt, fields_of_study)
                    processed_results.extend(processed_result)
                else:
                    # Handle cases where the API might return None without an exception
                    logging.warning(f"Task for source '{source}' ({value}) returned None.")

        else:
            logging.warning("No initial query criteria (DOI, Title, Topic) provided.")
        
        return processed_results
