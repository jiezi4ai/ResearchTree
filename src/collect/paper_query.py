import asyncio
from typing import List, Dict, Optional, Union, Literal, Tuple
from semanticscholar import AsyncSemanticScholar, Paper

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
        self.s2 = AsyncSemanticScholar() 

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


    def post_process(
            self, 
            source: Literal['title', 'doi'], 
            s2_result: Tuple[List[Paper.Paer], List[str]], 
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
    ):
        """Post process paper metadata"""
        # mark paper information based on source
        if source == 'doi' and isinstance(s2_result, tuple):
            papers_meata, not_found_dois = s2_result[0], s2_result[1]

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

        # reset id and filter papers 
        filtered_s2_paper_metadata = reset_and_filter_paper(
                    s2_paper_metadata=s2_paper_metadata,
                    from_dt=from_dt, 
                    to_dt=to_dt, 
                    fields_of_study=fields_of_study)
        
        # convert paper to json format (compatible with graph)
        processed = process_paper_metadata(filtered_s2_paper_metadata)

        return processed


    async def get_paper_info(
            self,
            seed_paper_titles: Optional[Union[List[str], str]] = None,
            seed_paper_dois: Optional[Union[List[str], str]] = None,
            limit: Optional[int] = 100,
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
    ):
        """Retrieve papers based on user's input paper titles or paper dois asynchronously."""
        if isinstance(seed_paper_titles, str):
            seed_paper_titles = [seed_paper_titles]
        
        if isinstance(seed_paper_dois, str):
            seed_paper_dois = [seed_paper_dois]

        tasks_with_source = []
        task_coroutines = []
        
        # Task for searching by DOIs
        if seed_paper_dois:
            logging.info(f"Fetching papers by {len(seed_paper_dois)} DOIs...")
            # return (papers, not_found_ids)
            coro = self.s2.get_papers(
                paper_ids=seed_paper_dois, 
                return_not_found=True)
            tasks_with_source.append({'source':'doi', 'values': seed_paper_dois, 'result': coro})
            task_coroutines.append(coro)
        
        if seed_paper_titles:
            for title in seed_paper_titles:
                logging.info(f"Fetching papers by title: '{title}...'")
                # return one single Paper object
                coro = self.s2.search_paper(
                    query=title, 
                    fields_of_study=fields_of_study, 
                    limit=limit, 
                    match_title=True
                    )
                tasks_with_source.append({'source':'title', 'values':[title], 'result': coro})
                task_coroutines.append(coro)
        
        # Run all initial query tasks concurrently
        all_results = []
        if task_coroutines:
            logging.info(f"Running {len(task_coroutines)} initial query tasks concurrently...")
            results = await asyncio.gather(*task_coroutines, return_exceptions=True)
            all_results.extend(results)

        # post-process output
        processed_results = {
            'fails': {
                'doi': {},  # k-v dict, where key is doi value, and value is for failed reason
                'title': {}, # k-v dict, where key is title value, and value is for failed reason
            },
            'success':{
                'paper_json': [],
                'excluded_papers': [],
                'not_found_dois': set(),
                'not_found_titles': set(),
            }
        }
        if len(all_results) > 0:
            for i, task_info in enumerate(tasks_with_source):
                source = task_info.get('source')
                values = task_info.get('values')
                result = all_results[i]

                # if fails
                if isinstance(result, Exception):
                    logging.error(f"Task for source '{source}' ({values}) failed: {result}", exc_info=result)
                    for val in values:
                        processed_results['fails'][source][val] = result

                # if success
                elif result is not None:
                    processed_result = self.post_process(source, result, from_dt, to_dt, fields_of_study)
                    
                    processed_results.extend(processed_result)
                else:
                    # Handle cases where the API might return None without an exception
                    logging.warning(f"Task for source '{source}' ({values}) returned None.")
        else:
            logging.warning("No initial query criteria (DOI, Title, Topic) provided.")
        
        return processed_results
