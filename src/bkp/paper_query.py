import asyncio
from typing import List, Dict, Optional, Union, Tuple
from semanticscholar.Paper import Paper

from apis.s2_api import SemanticScholarKit
from apis.s2_data_process import process_paper_data

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PaperQuery:
    def __init__(
            self,
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
    ):
        # initiate semantic scholar
        self.s2 = SemanticScholarKit() 
         
        # Filters
        self.from_dt = from_dt
        self.to_dt = to_dt
        self.fields_of_study = fields_of_study


    async def get_paper_info(
            self,
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
        if isinstance(seed_paper_titles, 'str'):
            seed_paper_titles = [seed_paper_titles]

        if isinstance(seed_paper_dois, 'str'):
            seed_paper_dois = [seed_paper_dois]

        tasks_with_source = []
        task_coroutines = []
        
        # Task for searching by DOIs
        if seed_paper_dois:  # doi search returns list of papers
            logging.info(f"Fetching papers by {len(seed_paper_dois)} DOIs...")
            coro = self.s2.get_papers(paper_ids=seed_paper_dois)
            tasks_with_source.append({'source':'doi', 'values': seed_paper_dois, 'result': coro})
            task_coroutines.append(coro)
        
        if seed_paper_titles: 
            for title in seed_paper_titles:  # title search returns a list of one single paper
                logging.info(f"Fetching papers by title: '{title}...'")
                coro = self.s2.search_paper(query=title, fields_of_study=fields_of_study, limit=limit, match_title=True)
                tasks_with_source.append({'source':'title', 'values':[title], 'result': coro})
                task_coroutines.append(coro)

        # Run all initial query tasks concurrently
        if task_coroutines:
            logging.info(f"Running {len(task_coroutines)} initial query tasks concurrently...")
            results = await asyncio.gather(*task_coroutines, return_exceptions=True)

        # post-process output
        s2_papers_metadata = []
        not_found_dois, not_found_titles = set(), set()
        if results is not None and len(results) > 0:
            for i, task_info in enumerate(tasks_with_source):
                source = task_info.get('source')
                values = task_info.get('values')
                result = results[i]
                # fails
                if isinstance(result, Exception):
                    logging.error(f"Task for source '{source}' ({values}) failed: {result}", exc_info=result)
                    if source == 'doi':
                        not_found_dois.update(values)
                    else:
                        not_found_titles.update(values)
                # success
                elif result is not None:
                    s2_papers_metadata.extend(result)
                # None
                else:
                    # Handle cases where the API might return None without an exception
                    logging.warning(f"Task for source '{source}' ({values}) returned None.")
                    if source == 'doi':
                        not_found_dois.update(values)
                    else:
                        not_found_titles.update(values)

        else:
            logging.warning("No initial query criteria (DOI, Title, Topic) provided.")

        s2_papers_metadata = [item.raw_data for item in s2_papers_metadata if isinstance(item, Paper)]

        processed_results = process_paper_data(s2_papers_metadata, from_dt, to_dt, fields_of_study)
        # add not found information
        processed_results['not_found_info'] = {}
        processed_results['not_found_info']['dois'] = not_found_dois
        processed_results['not_found_info']['title'] = not_found_titles

        return processed_results
