import math
import asyncio
from typing import List, Dict, Optional, Any, Callable
from semanticscholar import SemanticScholar, Paper, Author

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAX_CNT = 100  # set limit max
BATCH_SIZE = 100  # S2 API batch size restriction
DEFAULT_MAX_CONCURRENCY = 20 # Default concurrency limit
DEFAULT_SLEEP_INTERVAL = 3.0 # Default sleep interval in seconds

class SemanticScholarKit:
    def __init__(
        self,
        ss_api_key: str = None,
        ss_api_url: str = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY, # Added concurrency control
        sleep_interval: float = DEFAULT_SLEEP_INTERVAL, # Added sleep control
    ):
        """
        Initializes the SemanticScholarKit.

        :param str ss_api_key: (optional) private API key.
        :param str ss_api_url: (optional) custom API url.
        :param int max_concurrency: Maximum number of concurrent API calls allowed.
        :param float sleep_interval: Seconds to sleep after each API call (non-blocking).
        """
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be a positive integer")
        if sleep_interval < 0:
            raise ValueError("sleep_interval must be non-negative")

        self.scholar = SemanticScholar(api_key=ss_api_key, api_url=ss_api_url)
        self.batch_size = BATCH_SIZE
        self.max_cnt = MAX_CNT
        # --- Concurrency and Sleep Control ---
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.sleep_interval = sleep_interval
        logger.info(f"SemanticScholarKit initialized with max_concurrency={max_concurrency}, sleep_interval={sleep_interval}s")
        # --- End Concurrency and Sleep Control ---

    async def _execute_sync_with_controls(self, sync_func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Executes a synchronous function in a thread pool, managed by a semaphore
        for concurrency control, and adds a non-blocking sleep afterwards.

        :param sync_func: The synchronous function to execute (e.g., self._sync_get_papers).
        :param args: Positional arguments for the sync_func.
        :param kwargs: Keyword arguments for the sync_func.
        :return: The result of the sync_func.
        """
        async with self.semaphore:
            func_name = sync_func.__name__
            logger.debug(f"Semaphore acquired for {func_name}. Executing...")
            try:
                # Run the synchronous blocking function in a separate thread
                result = await asyncio.to_thread(sync_func, *args, **kwargs)
                logger.debug(f"Execution of {func_name} completed. Result type: {type(result)}. Sleeping for {self.sleep_interval}s.")
                # Perform non-blocking sleep *after* the sync function completes
                await asyncio.sleep(self.sleep_interval)
                return result
            except Exception as e:
                # Log the error from the sync execution
                logger.error(f"Exception during controlled execution of {func_name}: {e}", exc_info=True)
                # Still sleep even if an error occurred? Optional, but can prevent hammering on errors.
                logger.debug(f"Sleeping for {self.sleep_interval}s after error in {func_name}.")
                await asyncio.sleep(self.sleep_interval)
                # Re-raise the exception so asyncio.gather can capture it if needed,
                # or return a default value (like []) if preferred. Re-raising is often cleaner.
                raise e # Or return [] or appropriate default based on sync_func's error return
            finally:
                # Semaphore is released automatically by 'async with'
                logger.debug(f"Semaphore released for {func_name}.")


    # --- Synchronous Helper Methods (Unchanged, called via _execute_sync_with_controls) ---

    def _sync_get_papers(self, paper_ids: List[str]) -> List[Dict]:
        logger.info(f"_sync_get_papers: Thread started for batch ({len(paper_ids)} IDs, first 5: {paper_ids[:5]}...).")
        try:
            # The actual API call happens here
            batch_results: List[Paper] = self.scholar.get_papers(paper_ids=paper_ids)
            processed_results = [item._data for item in batch_results if hasattr(item, '_data')]
            logger.info(f"_sync_get_papers: API call successful for batch (first 5: {paper_ids[:5]}...), returning {len(processed_results)} items.")
            return processed_results
        except Exception as e:
            logger.error(f"Error in _sync_get_papers for batch (first 5 IDs: {paper_ids[:5]}...): {e}", exc_info=True)
            return [] # Original behavior: return empty list on error

    def _sync_get_authors(self, author_ids: List[str]) -> List[Dict]:
        logger.info(f"_sync_get_authors: Thread started for batch ({len(author_ids)} IDs, first 5: {author_ids[:5]}...).")
        try:
            # The actual API call happens here
            batch_results: List[Author] = self.scholar.get_authors(author_ids=author_ids)
            processed_results = [item._data for item in batch_results if hasattr(item, '_data')]
            logger.info(f"_sync_get_authors: API call successful for batch (first 5: {author_ids[:5]}...), returning {len(processed_results)} items.")
            return processed_results
        except Exception as e:
            logger.error(f"Error in _sync_get_authors for batch (first 5 IDs: {author_ids[:5]}...): {e}", exc_info=True)
            return [] # Original behavior: return empty list on error

    def _sync_search_paper_by_keywords(self, **kwargs) -> List[Dict]:
        query = kwargs.get('query', 'N/A')
        limit = kwargs.get('limit', self.max_cnt)
        match_title = kwargs.get('match_title', False)
        logger.info(f"_sync_search_paper_by_keywords: Thread started for query '{query[:50]}...' with limit {limit}.")
        try:
            # The actual API call happens here
            results = self.scholar.search_paper(**kwargs)
            paper_metadata = []
            if match_title == True and isinstance(results, Paper.Paper):  # single paper match
                paper_metadata.append(results._data)
            elif results and hasattr(results, 'total') and results.total > 0:
                count = 0
                # Iterate through results safely, respecting limit passed in kwargs
                for item in results:
                    if count >= limit: # Use limit from kwargs
                        break
                    if hasattr(item, '_data'):
                        paper_metadata.append(item._data)
                    count += 1
                logger.info(f"_sync_search_paper_by_keywords: API call successful for query '{query[:50]}...', returning {len(paper_metadata)} items.")
            else:
                 logger.info(f"_sync_search_paper_by_keywords: No results found for query '{query[:50]}...'.")
            return paper_metadata
        except Exception as e:
            logger.error(f"Error in _sync_search_paper_by_keywords for query '{query[:50]}...': {e}", exc_info=True)
            return []

    def _sync_get_paper_references(self, paper_id: str, limit: int) -> List[Dict]:
        logger.info(f"_sync_get_paper_references: Thread started for paper {paper_id} with limit {limit}.")
        try:
            # The actual API call happens here
            results = self.scholar.get_paper_references(paper_id, limit=limit)
            processed_results = [item._data for item in results if hasattr(item, '_data')]
            logger.info(f"_sync_get_paper_references: API call successful for paper {paper_id}, returning {len(processed_results)} items.")
            return processed_results
        except Exception as e:
            logger.error(f"Error in _sync_get_paper_references for {paper_id}: {e}", exc_info=True)
            return [] # Original behavior: return empty list on error

    def _sync_get_paper_citations(self, paper_id: str, limit: int) -> List[Dict]:
        logger.info(f"_sync_get_paper_citations: Thread started for paper {paper_id} with limit {limit}.")
        try:
            # The actual API call happens here
            results = self.scholar.get_paper_citations(paper_id, limit=limit)
            processed_results = [item._data for item in results if hasattr(item, '_data')]
            logger.info(f"_sync_get_paper_citations: API call successful for paper {paper_id}, returning {len(processed_results)} items.")
            return processed_results
        except Exception as e:
            logger.error(f"Error in _sync_get_paper_citations for {paper_id}: {e}", exc_info=True)
            return [] # Original behavior: return empty list on error

    def _sync_get_recommended_papers(self, positive_paper_ids: List[str], negative_paper_ids: Optional[List[str]], limit: int) -> List[Dict]:
        logger.info(f"_sync_get_recommended_papers: Thread started based on {len(positive_paper_ids)} positive IDs (first 5: {positive_paper_ids[:5]}...) with limit {limit}.")
        try:
            # The actual API call happens here
            results = self.scholar.get_recommended_papers_from_lists(
                positive_paper_ids=positive_paper_ids,
                negative_paper_ids=negative_paper_ids,
                limit=limit
            )
            processed_results = [item._data for item in results if hasattr(item, '_data')]
            logger.info(f"_sync_get_recommended_papers: API call successful, returning {len(processed_results)} items.")
            return processed_results
        except Exception as e:
            logger.error(f"Error in _sync_get_recommended_papers regarding positive IDs (first 5: {positive_paper_ids[:5]}...): {e}", exc_info=True)
            return [] # Original behavior: return empty list on error


    # --- Asynchronous Public Methods (Interface Unchanged, Logic Updated) ---
    async def async_search_paper_by_ids(
        self,
        id_list: List[str]
    ) -> List[Dict]:
        """Search paper by id asynchronously, respecting concurrency and sleep controls."""
        valid_id_list = [x for x in id_list if x and isinstance(x, str)]
        id_cnt = len(valid_id_list)
        paper_metadata = []

        if id_cnt > 0:
            batch_size = self.batch_size
            batch_cnt = math.ceil(id_cnt / batch_size)
            batches = [valid_id_list[i * batch_size:(i + 1) * batch_size] for i in range(batch_cnt)]

            tasks = []
            logger.info(f"async_search_paper_by_ids: Creating {len(batches)} tasks for {id_cnt} IDs.")
            for batch in batches:
                # Use the controlled execution wrapper for each batch task
                tasks.append(self._execute_sync_with_controls(self._sync_get_papers, batch))

            logger.info(f"async_search_paper_by_ids: Gathering {len(tasks)} tasks...")
            try:
                # return_exceptions=True allows gather to complete even if some tasks fail
                batch_results_list = await asyncio.gather(*tasks, return_exceptions=True)
                logger.info(f"async_search_paper_by_ids: Gather complete. Processing results.")

                # Aggregate results, handling potential exceptions from gather
                for result in batch_results_list:
                    if isinstance(result, Exception):
                        # Error was already logged in _execute_sync_with_controls or the sync method
                        logger.error(f"A batch task for async_search_paper_by_ids failed: {result}")
                        # Decide how to handle partial failures (e.g., log, skip)
                        # Current behavior: skip failed batches
                    elif isinstance(result, list):
                        paper_metadata.extend(result)
                    else:
                        # Should not happen if _execute_sync_with_controls handles errors properly
                        logger.warning(f"Unexpected result type {type(result)} from batch task: {result}")

            except Exception as e:
                # Catch potential errors in asyncio.gather itself (less common)
                logger.error(f"async_search_paper_by_ids: Exception during asyncio.gather: {e}", exc_info=True)
                paper_metadata = [] # Reset results on major failure

        return paper_metadata

    async def async_search_author_by_ids(
        self,
        author_ids: List[str],
        with_abstract: Optional[bool] = False
    ) -> List[Dict]:
        """Search author by ids asynchronously, respecting concurrency and sleep controls."""
        valid_id_list = [x for x in author_ids if x and isinstance(x, str)]
        id_cnt = len(valid_id_list)
        author_metadata = []

        if id_cnt > 0:
            batch_size = self.batch_size
            batch_cnt = math.ceil(id_cnt / batch_size)
            batches = [valid_id_list[i * batch_size:(i + 1) * batch_size] for i in range(batch_cnt)]
            logger.info(f"async_search_author_by_ids: Fetching {id_cnt} authors by ID in {batch_cnt} batches.")

            tasks = []
            for batch in batches:
                 # Use the controlled execution wrapper
                tasks.append(self._execute_sync_with_controls(self._sync_get_authors, batch))

            logger.info(f"async_search_author_by_ids: Gathering {len(tasks)} tasks...")
            batch_results_list = await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(f"async_search_author_by_ids: Gather complete. Processing results.")

            for result in batch_results_list:
                if isinstance(result, Exception):
                    logger.error(f"A batch task for async_search_author_by_ids failed: {result}")
                    # Skip failed batches
                elif isinstance(result, list):
                    author_metadata.extend(result)
                else:
                     logger.warning(f"Unexpected result type {type(result)} from author batch task: {result}")

        # --- Handle with_abstract asynchronously (logic unchanged, but uses modified async_search_paper_by_ids) ---
        if with_abstract and author_metadata:
            papers_missing_abstract_ids = set()
            for info in author_metadata:
                papers = info.get('papers', [])
                for paper in papers:
                    paper_id = paper.get('paperId')
                    abstract = paper.get('abstract')
                    if paper_id and abstract is None:
                        papers_missing_abstract_ids.add(paper_id)

            if papers_missing_abstract_ids:
                logger.info(f"Fetching abstracts for {len(papers_missing_abstract_ids)} papers from authors (using controlled search).")
                # This call will now also respect the concurrency/sleep settings internally
                paper_metadata_with_abstracts = await self.async_search_paper_by_ids(
                    id_list=list(papers_missing_abstract_ids)
                )
                ref_abstracts = {
                    item['paperId']: item['abstract']
                    for item in paper_metadata_with_abstracts
                    if item.get('paperId') and item.get('abstract')
                }

                if ref_abstracts:
                    logger.info(f"Found abstracts for {len(ref_abstracts)} papers.")
                    for info in author_metadata:
                        papers = info.get('papers', [])
                        for paper in papers:
                            paper_id = paper.get('paperId')
                            if paper_id in ref_abstracts and paper.get('abstract') is None:
                                paper['abstract'] = ref_abstracts[paper_id]
                else:
                    logger.warning("Secondary search for author abstracts did not find any missing abstracts.")

        return author_metadata

    async def async_search_paper_by_keywords(
        self,
        query: str,
        year: str = None,
        publication_types: list = None,
        open_access_pdf: bool = None,
        venue: list = None,
        fields_of_study: list = None,
        publication_date_or_year: str = None,
        min_citation_count: int = None,
        limit: int = 100,
        bulk: bool = False, # Note: 'bulk' might require special handling/pagination not shown here
        sort: str = None,
        match_title: bool = False
    ) -> List[Dict]:
        """Search for papers by keyword asynchronously, respecting concurrency and sleep controls."""
        search_kwargs = {
            "query": query,
            "year": year,
            "publication_types": publication_types,
            "open_access_pdf": open_access_pdf,
            "venue": venue,
            "fields_of_study": fields_of_study,
            "publication_date_or_year": publication_date_or_year,
            "min_citation_count": min_citation_count,
            "limit": min(limit, self.max_cnt), # Apply overall max limit
            "bulk": bulk,
            "sort": sort,
            "match_title": match_title,
        }
        # Remove None values as the underlying library might not handle them well
        search_kwargs = {k: v for k, v in search_kwargs.items() if v is not None}

        logger.info(f"async_search_paper_by_keywords: Searching papers by keyword: '{query[:50]}...' with effective limit {search_kwargs.get('limit')}.")
        try:
             # Use the controlled execution wrapper
            paper_metadata = await self._execute_sync_with_controls(self._sync_search_paper_by_keywords, **search_kwargs)
        except Exception as e:
             # Error already logged in wrapper, return default
             logger.error(f"async_search_paper_by_keywords: Failed for query '{query[:50]}...': {e}")
             paper_metadata = [] # Return empty list on failure

        return paper_metadata

    async def async_get_s2_cited_papers( # Corresponds to get_paper_references
        self,
        paper_id: str,
        limit: int = 100,
        with_abstract: Optional[bool] = False
    ) -> List[Dict]:
        """Get papers cited by this paper (references) asynchronously, respecting controls."""
        max_limit = min(limit, self.max_cnt)
        logger.info(f"async_get_s2_cited_papers: Fetching references for paper {paper_id} with effective limit {max_limit}.")

        refs_metadata = []
        try:
            # Use the controlled execution wrapper
            refs_metadata = await self._execute_sync_with_controls(self._sync_get_paper_references, paper_id, max_limit)
        except Exception as e:
             logger.error(f"async_get_s2_cited_papers: Failed for paper {paper_id}: {e}")
             refs_metadata = [] # Return empty list on failure


        # --- Handle with_abstract asynchronously (logic unchanged) ---
        if with_abstract and refs_metadata:
            papers_missing_abstract_ids = set()
            for info in refs_metadata:
                paper = info.get('citedPaper') # Key is 'citedPaper'
                if isinstance(paper, dict):
                    p_id = paper.get('paperId')
                    abstract = paper.get('abstract')
                    if p_id and abstract is None:
                        papers_missing_abstract_ids.add(p_id)

            if papers_missing_abstract_ids:
                logger.info(f"Fetching abstracts for {len(papers_missing_abstract_ids)} cited papers (references).")
                # This call will now also respect the concurrency/sleep settings
                paper_metadata_with_abstracts = await self.async_search_paper_by_ids(id_list=list(papers_missing_abstract_ids))
                ref_abstracts = {
                    item['paperId']: item['abstract']
                    for item in paper_metadata_with_abstracts
                    if item.get('paperId') and item.get('abstract')
                }

                if ref_abstracts:
                    logger.info(f"Found abstracts for {len(ref_abstracts)} cited papers.")
                    for info in refs_metadata:
                        paper = info.get('citedPaper')
                        if isinstance(paper, dict):
                           p_id = paper.get('paperId')
                           if p_id in ref_abstracts and paper.get('abstract') is None:
                               paper['abstract'] = ref_abstracts[p_id]
                else:
                    logger.warning("Secondary search did not find any missing abstracts for cited papers.")

        return refs_metadata

    async def async_get_s2_citing_papers( # Corresponds to get_paper_citations
        self,
        paper_id: str,
        limit: int = 100,
        with_abstract: Optional[bool] = False
    ) -> List[Dict]:
        """Get papers citing this paper asynchronously, respecting controls."""
        max_limit = min(limit, self.max_cnt) # Apply overall max limit
        logger.info(f"async_get_s2_citing_papers: Fetching citations for paper {paper_id} with effective limit {max_limit}.")

        citedby_metadata = []
        try:
             # Use the controlled execution wrapper
            citedby_metadata = await self._execute_sync_with_controls(self._sync_get_paper_citations, paper_id, max_limit)
        except Exception as e:
            logger.error(f"async_get_s2_citing_papers: Failed for paper {paper_id}: {e}")
            citedby_metadata = [] # Return empty list on failure

        # --- Handle with_abstract asynchronously (logic unchanged) ---
        if with_abstract and citedby_metadata:
            papers_missing_abstract_ids = set()
            for info in citedby_metadata:
                paper = info.get('citingPaper') # Note: key is 'citingPaper'
                if isinstance(paper, dict):
                    p_id = paper.get('paperId')
                    abstract = paper.get('abstract')
                    if p_id and abstract is None:
                        papers_missing_abstract_ids.add(p_id)

            if papers_missing_abstract_ids:
                logger.info(f"Fetching abstracts for {len(papers_missing_abstract_ids)} citing papers.")
                # This call will now also respect the concurrency/sleep settings
                paper_metadata_with_abstracts = await self.async_search_paper_by_ids(id_list=list(papers_missing_abstract_ids))
                ref_abstracts = {
                    item['paperId']: item['abstract']
                    for item in paper_metadata_with_abstracts
                    if item.get('paperId') and item.get('abstract')
                }

                if ref_abstracts:
                    logger.info(f"Found abstracts for {len(ref_abstracts)} citing papers.")
                    for info in citedby_metadata:
                        paper = info.get('citingPaper')
                        if isinstance(paper, dict):
                            p_id = paper.get('paperId')
                            if p_id in ref_abstracts and paper.get('abstract') is None:
                                paper['abstract'] = ref_abstracts[p_id]
                else:
                    logger.warning("Secondary search did not find any missing abstracts for citing papers.")

        return citedby_metadata

    async def async_get_s2_recommended_papers(
        self,
        positive_paper_ids: List[str],
        negative_paper_ids: List[str] = None,
        limit: int = 100,
        with_abstract: Optional[bool] = False
    ) -> List[Dict]:
        """Get recommended papers asynchronously, respecting controls."""
        max_limit = min(limit, self.max_cnt) # Apply overall max limit
        logger.info(f"async_get_s2_recommended_papers: Fetching recommendations based on {len(positive_paper_ids)} positive IDs with effective limit {max_limit}.")

        rec_metadata = []
        try:
            # Use the controlled execution wrapper
            rec_metadata = await self._execute_sync_with_controls(
                self._sync_get_recommended_papers,
                positive_paper_ids,
                negative_paper_ids,
                max_limit
            )
        except Exception as e:
            logger.error(f"async_get_s2_recommended_papers: Failed for positive IDs (first 5: {positive_paper_ids[:5]}...): {e}")
            rec_metadata = [] # Return empty list on failure


        # --- Handle with_abstract asynchronously (logic unchanged) ---
        if with_abstract and rec_metadata:
            papers_missing_abstract_ids = set()
            for paper in rec_metadata: # Recommended papers are directly in the list
                paper_id = paper.get('paperId')
                abstract = paper.get('abstract')
                if paper_id and abstract is None:
                    papers_missing_abstract_ids.add(paper_id)

            if papers_missing_abstract_ids:
                logger.info(f"Fetching abstracts for {len(papers_missing_abstract_ids)} recommended papers.")
                 # This call will now also respect the concurrency/sleep settings
                paper_metadata_with_abstracts = await self.async_search_paper_by_ids(id_list=list(papers_missing_abstract_ids),)
                ref_abstracts = {
                    item['paperId']: item['abstract']
                    for item in paper_metadata_with_abstracts
                    if item.get('paperId') and item.get('abstract')
                }

                if ref_abstracts:
                    logger.info(f"Found abstracts for {len(ref_abstracts)} recommended papers.")
                    for paper in rec_metadata:
                        paper_id = paper.get('paperId')
                        if paper_id in ref_abstracts and paper.get('abstract') is None:
                            paper['abstract'] = ref_abstracts[paper_id]
                else:
                    logger.warning("Secondary search did not find any missing abstracts for recommended papers.")

        return rec_metadata