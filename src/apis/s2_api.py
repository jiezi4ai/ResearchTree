# Potential update
# disable internal retry
# shorten time out

import math
import asyncio
import logging
from tenacity import RetryError
from typing import List, Tuple, Union, Optional, Set, Any, Literal

# Import semantic scholar
from semanticscholar import AsyncSemanticScholar 
from semanticscholar.Paper import Paper
from semanticscholar.Author import Author
from semanticscholar.PaginatedResults import PaginatedResults
from semanticscholar.Autocomplete import Autocomplete
from semanticscholar.BaseReference import BaseReference # Needed for type hints
from semanticscholar.Citation import Citation         # Needed for type hints
from semanticscholar.Reference import Reference         # Needed for type hints
from semanticscholar.SemanticScholarException import ObjectNotFoundException, BadQueryParametersException

# Import aiohttp exception for status code checking if ApiRequester uses it
try:
    import aiohttp
    HttpClientError = aiohttp.ClientResponseError
except ImportError:
    # Fallback if aiohttp is not directly used or installed
    HttpClientError = Exception 
    print("Warning: aiohttp not found. Status code 429 check might be inaccurate.")


DEFAULT_TIME_OUT = 30
DEFAULT_BTCH_SIZE = 100
DEFAULT_MAX_CONCURRENCY = 20  # Default concurrency limit
DEFAULT_SLEEP_INTERVAL = 3.0  # Default sleep interval in seconds
DEFAULT_MAX_RETRIES = 20  # Default max retries if failed
ENABLE_INTERNAL_RETRIES = False  # Disable internal retries



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



class SemanticScholarKit:
    """
    A wrapper class for AsyncSemanticScholar providing enhanced error handling,
    retry mechanisms for rate limiting (429 errors), concurrency control,
    and a helper to fetch all items from paginated results asynchronously.
    """

    def __init__(
        self,
        # AsyncSemanticScholar parameters
        timeout: int = 30,
        api_key: str = None,
        api_url: str = None,
        # Kit parameters
        batch_size: int = DEFAULT_BTCH_SIZE,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        max_retry: int = DEFAULT_MAX_RETRIES,
        sleep_interval: float = DEFAULT_SLEEP_INTERVAL, 
        log_level: int = logging.INFO,
        # Pass underlying retry setting to AsyncSemanticScholar's requester
        internal_requester_retry: bool = ENABLE_INTERNAL_RETRIES 
    ) -> None:
        """
        Initializes the SemanticScholarKit.

        :param int timeout: Timeout for individual requests (passed to AsyncSemanticScholar).
        :param str api_key: Private API key (passed to AsyncSemanticScholar).
        :param str api_url: Custom API URL (passed to AsyncSemanticScholar).
        :param int max_concurrency: Maximum number of concurrent requests allowed.
        :param int max_retry: Maximum number of retries specifically for 429 errors.
        :param float sleep_interval: Seconds to wait before retrying after a 429 error.
        :param int log_level: Logging level for the kit (e.g., logging.INFO, logging.WARNING).
        :param bool internal_requester_retry: Whether to enable the retry mechanism within
               AsyncSemanticScholar's internal ApiRequester (recommended).
        """
        logger.setLevel(log_level)
        
        self.s2 = AsyncSemanticScholar(
            timeout=timeout,
            api_key=api_key,
            api_url=api_url,
            retry=internal_requester_retry 
        )
        
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be greater than 0")
        if max_retry < 0:
            raise ValueError("max_retry must be non-negative")
        if sleep_interval <= 0:
            raise ValueError("sleep_interval must be positive")
            
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.max_retry = max_retry
        self.sleep_interval = sleep_interval
        
        self.semaphore = asyncio.Semaphore(self.max_concurrency)
        self.not_found_ids: Set[str] = set()

        logger.info(f"SemanticScholarKit initialized with: "
                    f"max_concurrency={max_concurrency}, "
                    f"max_retry={max_retry}, "
                    f"sleep_interval={sleep_interval}s")

    async def _execute_with_retry(self, coro_func, *args, **kwargs) -> Any:
        """Internal helper to execute a coroutine with retry and error handling."""
        identifier = None
        if 'paper_id' in kwargs: identifier = kwargs['paper_id']
        elif 'author_id' in kwargs: identifier = kwargs['author_id']
        elif 'paper_ids' in kwargs: identifier = f"batch:{len(kwargs['paper_ids'])} papers"
        elif 'author_ids' in kwargs: identifier = f"batch:{len(kwargs['author_ids'])} authors"
        elif 'query' in kwargs: identifier = f"query:'{kwargs.get('query', '')[:30]}...'" # Use get for safety
        elif 'url' in kwargs and 'parameters' in kwargs: # For direct requester calls
             identifier = f"url:{kwargs['url']}?{kwargs['parameters']}"
        
        # Try to get the original function name if available
        func_name = getattr(coro_func, '__name__', 'unknown_coroutine') 

        async with self.semaphore:
            for attempt in range(self.max_retry + 1):
                try:
                    result = await coro_func(*args, **kwargs)
                    return result
                
                except RetryError as e: # Catch tenacity error first
                    # Check if the root cause is the 429 ConnectionRefusedError
                    is_429 = False
                    if isinstance(e.__cause__, ConnectionRefusedError):
                        cause_str = str(e.__cause__).lower()
                        if "429" in cause_str or "too many requests" in cause_str:
                            is_429 = True

                    if is_429:
                        if attempt < self.max_retry:
                            logger.warning(
                                f"Rate limit (429) hit (via RetryError) for {func_name} "
                                f"(ID/Query/URL: {identifier}). Underlying tenacity retries failed. "
                                f"Attempting outer retry {attempt + 1}/{self.max_retry + 1}. "
                                f"Retrying after {self.sleep_interval} seconds..."
                            )
                            await asyncio.sleep(self.sleep_interval)
                            continue
                        else:
                            logger.error(
                                f"Rate limit (429) hit (via RetryError) for {func_name} "
                                f"(ID/Query/URL: {identifier}). Max outer retries ({self.max_retry}) exceeded after underlying tenacity retries failed."
                            )
                            raise e # Re-raise RetryError
                    else:
                        # If RetryError was caused by something else, treat as unexpected
                        logger.exception(f"A tenacity RetryError occurred for {func_name} "
                                        f"(ID/Query/URL: {identifier}). Cause: {e.__cause__}. "
                                        f"Attempt {attempt + 1}/{self.max_retry + 1}.")
                        raise e # Re-raise RetryError

                except ConnectionRefusedError as e: # Keep this for cases tenacity might not catch (unlikely for 429 with this library)
                    is_429 = False
                    if hasattr(e, 'status') and e.status == 429:
                        is_429 = True
                    else:
                        e_str = str(e).lower()
                        if "429" in e_str or "too many requests" in e_str:
                            is_429 = True

                    if is_429:
                        if attempt < self.max_retry:
                            logger.warning(
                                f"Rate limit (429) hit directly for {func_name} "
                                f"(ID/Query/URL: {identifier}). Attempt {attempt + 1}/{self.max_retry + 1}. "
                                f"Retrying after {self.sleep_interval} seconds..."
                            )
                            await asyncio.sleep(self.sleep_interval)
                            continue
                        else:
                            logger.error(
                                f"Rate limit (429) hit directly for {func_name} "
                                f"(ID/Query/URL: {identifier}). Max retries ({self.max_retry}) exceeded."
                            )
                            raise e
                    else:
                        # Other ConnectionRefusedError
                        logger.error(f"ConnectionRefusedError (not 429) calling {func_name} "
                                    f"(ID/Query/URL: {identifier}): {e}")
                        raise e

                except ObjectNotFoundException as e:
                    # ... (your existing logic) ...
                    if func_name in ['get_paper', 'get_author']:
                        return None
                    raise e

                except BadQueryParametersException as e:
                     logger.error(f"BadQueryParametersException for {func_name} "
                                  f"(ID/Query/URL: {identifier}): {e}")
                     raise e 

                except Exception as e:
                    logger.exception(f"An unexpected error occurred calling {func_name} "
                                     f"(ID/Query/URL: {identifier}). Attempt {attempt + 1}/{self.max_retry + 1}.")
                    raise e 

            logger.error(f"Failed to execute {func_name} after {self.max_retry + 1} attempts (should not be reached).")
            return None # Fallback

    # --- Helper for Paginated Results ---
    async def get_all_paginated_items(self, results: PaginatedResults) -> List[Any]:
        """
        Asynchronously fetches all items from a PaginatedResults object.

        This works around the limitation of the synchronous iterator trying
        to call run_until_complete within an async context. It uses the kit's
        retry and concurrency mechanisms for subsequent page fetches.
        """
        if not results:
            return []
            
        all_items = list(results.items) # Start with the first page already fetched
        
        # Access internal state needed for subsequent fetches
        offset = results.offset + len(results.items)
        limit = results._limit
        total = results.total
        max_results = results._max_results 
        # Use the requester from the underlying s2 instance
        requester = self.s2._requester 
        data_type = results._data_type 
        auth_header = self.s2.auth_header # Use kit's auth header
        url = results._url
        query = results._query
        fields = results._fields

        # Store last fetched items count for short page check
        last_fetch_count = len(results.items) 

        logger.debug(f"Starting fetch_all: initial items={len(all_items)}, offset={offset}, limit={limit}, total={total}, max_results={max_results}")

        while True:
            # Check termination conditions
            if total is not None and offset >= total:
                logger.debug(f"Stopping fetch_all: offset {offset} >= total {total}")
                break
            if len(all_items) >= max_results: 
                logger.debug(f"Stopping fetch_all: item count {len(all_items)} >= max_results {max_results}")
                break
            # If the last fetch returned fewer items than the limit, we are done
            if offset > results.offset and last_fetch_count < limit: 
                 logger.debug(f"Stopping fetch_all: last fetch ({last_fetch_count}) was less than limit ({limit})")
                 break

            # Prepare parameters for the next request
            params_list = [f'offset={offset}', f'limit={limit}']
            if fields:
                fields_str = ','.join(fields)
                params_list.append(f'fields={fields_str}')
            
            # Prepend query if it exists (for search endpoints)
            # Note: ApiRequester likely handles URL encoding, pass query separately if needed by it
            current_query = query
            current_url = url
            if current_query:
                 # Assuming ApiRequester's get_data_async handles query in parameters string
                 params_list.insert(0, f'query={current_query}') # Add query param
            
            parameters = '&'.join(params_list)

            logger.debug(f"Fetching next page: url={current_url}, params={parameters}")
            try:
                # Use the kit's retry mechanism for the fetch itself
                next_page_data = await self._execute_with_retry(
                     requester.get_data_async, # Call the requester's async method directly
                     url=current_url, 
                     parameters=parameters, 
                     headers=auth_header,
                     payload=None # Assuming GET for pagination
                )
                
                if next_page_data is None:
                     logger.warning("fetch_all: _execute_with_retry returned None during pagination, stopping.")
                     break

                # Determine where the actual list of items is
                items_key = 'data' # Default for most graph endpoints
                if '/recommendations/' in current_url:
                    items_key = 'recommendedPapers'
                elif '/autocomplete' in current_url:
                    items_key = 'matches'
                
                if not next_page_data or items_key not in next_page_data or not next_page_data[items_key]:
                    logger.debug(f"Stopping fetch_all: No more data in '{items_key}' from API.")
                    break # No more data

                new_items_data = next_page_data[items_key]
                new_items = [data_type(item) for item in new_items_data]
                all_items.extend(new_items)
                
                last_fetch_count = len(new_items) # Update for next iteration's check
                offset += last_fetch_count

                # Update total if learned (usually in 'total' key for graph)
                if total is None and 'total' in next_page_data:
                     total = next_page_data['total']
                     logger.debug(f"Updated total to {total}")

                logger.debug(f"Fetched {last_fetch_count} items. Total items now: {len(all_items)}. Next offset: {offset}")

            except Exception as e:
                logger.exception(f"Error fetching subsequent page in get_all_paginated_items: {e}")
                break # Stop fetching on error

        # Ensure we don't exceed max_results if it was hit exactly
        return all_items[:max_results]

    # --- Wrapped AsyncSemanticScholar Methods ---

    async def get_paper(
        self,
        paper_id: str,
        fields: list = None
    ) -> Optional[Paper]:
        """ (Docstring unchanged) """
        return await self._execute_with_retry(
            self.s2.get_paper,
            paper_id=paper_id,
            fields=fields
        )

    async def get_papers( 
        self,
        paper_ids: List[str],
        fields: list = None,
        return_not_found: bool = False
    ) -> Union[List[Paper], Tuple[List[Paper], List[str]]]:
        """ (Docstring unchanged) """
        paper_metadata, not_found_ids = [], []

        paper_cnt = len(paper_ids)
        if paper_cnt <= 0: 
            if return_not_found:
                return [], []
            else:
                return []

        batch_size = self.batch_size
        batch_cnt = math.ceil(paper_cnt / batch_size)
        batches = [paper_ids[i * batch_size:(i + 1) * batch_size] for i in range(batch_cnt)]

        tasks = []
        logger.info(f"get_papers: Creating {len(batches)} tasks for {batch_cnt} IDs.")
        for batch in batches:
            tasks.append(self._execute_with_retry(
                self.s2.get_papers,
                paper_ids=batch,
                fields=fields,
                return_not_found=return_not_found
            ))
        
        logger.info(f"get_papers: Gathering {len(tasks)} tasks...")
        try:
            # return_exceptions=True allows gather to complete even if some tasks fail
            batch_results_list = await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(f"get_papers: Gather complete. Processing results.")

            # Aggregate results, handling potential exceptions from gather
            for idx, result in enumerate(batch_results_list):
                if isinstance(result, Exception):
                    # Error was already logged in _execute_sync_with_controls or the sync method
                    logger.error(f"A batch task for get_papers failed: {result}")
                    if return_not_found:
                        not_found_ids.extend(batches[idx])
                elif isinstance(result, list):
                    if return_not_found and isinstance(result, tuple):
                        papers_list = result[0] if result[0] is not None else []
                        paper_metadata.extend(papers_list)
                        not_found_ids.extend(result[1])
                    elif not return_not_found and isinstance(result, list):
                        paper_metadata.extend(result)
                else:
                    # Should not happen if _execute_sync_with_controls handles errors properly
                    logger.warning(f"Unexpected result type {type(result)} from batch task: {result}")
                    not_found_ids.extend(batches[idx])

        except Exception as e:
            # Catch potential errors in asyncio.gather itself (less common)
            logger.error(f"get_papers: Exception during asyncio.gather: {e}", exc_info=True)
            not_found_ids.extend(batches)
        
        if return_not_found:
            return paper_metadata, not_found_ids
        return paper_metadata


    async def get_paper_authors(
        self,
        paper_id: str,
        fields: list = None,
        limit: int = 100
    ) -> PaginatedResults:
        """ (Docstring unchanged) """
        # This returns the initial PaginatedResults object
        # Use get_all_paginated_items separately to fetch all items
        return await self._execute_with_retry(
            self.s2.get_paper_authors,
            paper_id=paper_id,
            fields=fields,
            limit=limit
        )
        # Note: Returning an empty PaginatedResults on failure might be misleading
        # if the caller expects to use get_all_paginated_items. Returning None might be better.
        # Let's stick to returning the result from _execute_with_retry for now.


    async def get_paper_citations(
        self,
        paper_id: str,
        fields: list = None,
        limit: int = 100
    ) -> PaginatedResults:
        """ (Docstring unchanged) """
        return await self._execute_with_retry(
            self.s2.get_paper_citations,
            paper_id=paper_id,
            fields=fields,
            limit=limit
        )

    async def get_paper_references(
        self,
        paper_id: str,
        fields: list = None,
        limit: int = 100
    ) -> PaginatedResults:
        """ (Docstring unchanged) """
        return await self._execute_with_retry(
            self.s2.get_paper_references,
            paper_id=paper_id,
            fields=fields,
            limit=limit
        )


    async def search_paper(
        self,
        query: str,
        year: str = None,
        publication_types: list = None,
        open_access_pdf: bool = None,
        venue: list = None,
        fields_of_study: list = None,
        fields: list = None,
        publication_date_or_year: str = None,
        min_citation_count: int = None,
        limit: int = 100,
        bulk: bool = False,
        sort: str = None,
        match_title: bool = False
    ) -> Union[PaginatedResults, Optional[Paper]]:
        """ (Docstring unchanged) """
        # Returns initial PaginatedResults or single Paper if match_title=True
        return await self._execute_with_retry(
            self.s2.search_paper,
            query=query, year=year, publication_types=publication_types,
            open_access_pdf=open_access_pdf, venue=venue,
            fields_of_study=fields_of_study, fields=fields,
            publication_date_or_year=publication_date_or_year,
            min_citation_count=min_citation_count, limit=limit,
            bulk=bulk, sort=sort, match_title=match_title
        )


    async def get_author(
        self,
        author_id: str,
        fields: list = None
    ) -> Optional[Author]:
        """ (Docstring unchanged) """
        return await self._execute_with_retry(
            self.s2.get_author,
            author_id=author_id,
            fields=fields
        )

    async def get_authors(
        self,
        author_ids: List[str],
        fields: list = None,
        return_not_found: bool = False
    ) -> Union[List[Author], Tuple[List[Author], List[str]]]:
        """ (Docstring unchanged) """
        author_metadata, not_found_ids = [], []

        author_cnt = len(author_ids)
        if author_cnt <= 0: 
            if return_not_found:
                return [], []
            else:
                return []

        batch_size = self.batch_size
        batch_cnt = math.ceil(author_cnt / batch_size)
        batches = [author_ids[i * batch_size:(i + 1) * batch_size] for i in range(batch_cnt)]

        tasks = []
        logger.info(f"get_authors: Creating {len(batches)} tasks for {batch_cnt} IDs.")
        for batch in batches:
            tasks.append(self._execute_with_retry(
                self.s2.get_authors,
                author_ids=author_ids,
                fields=fields,
                return_not_found=return_not_found
            ))
        
        logger.info(f"get_authors: Gathering {len(tasks)} tasks...")
        try:
            # return_exceptions=True allows gather to complete even if some tasks fail
            batch_results_list = await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(f"get_authors: Gather complete. Processing results.")

            # Aggregate results, handling potential exceptions from gather
            for idx, result in enumerate(batch_results_list):
                if isinstance(result, Exception):
                    # Error was already logged in _execute_sync_with_controls or the sync method
                    logger.error(f"A batch task for get_papers failed: {result}")
                    if return_not_found:
                        not_found_ids.extend(batches[idx])
                elif isinstance(result, list):
                    if return_not_found and isinstance(result, tuple):
                        authors_list = result[0] if result[0] is not None else []
                        author_metadata.extend(authors_list)
                        not_found_ids.extend(result[1])
                    elif not return_not_found and isinstance(result, list):
                        author_metadata.extend(result)
                else:
                    # Should not happen if _execute_sync_with_controls handles errors properly
                    logger.warning(f"Unexpected result type {type(result)} from batch task: {result}")
                    not_found_ids.extend(batches[idx])

        except Exception as e:
            # Catch potential errors in asyncio.gather itself (less common)
            logger.error(f"get_authors: Exception during asyncio.gather: {e}", exc_info=True)
            not_found_ids.extend(batches)
        
        if return_not_found:
            return author_metadata, not_found_ids
        return author_metadata
    

    async def get_author_papers(
        self,
        author_id: str,
        fields: list = None,
        limit: int = 100
    ) -> PaginatedResults:
        """ (Docstring unchanged) """
        return await self._execute_with_retry(
            self.s2.get_author_papers,
            author_id=author_id,
            fields=fields,
            limit=limit
        )


    async def search_author(
        self,
        query: str,
        fields: list = None,
        limit: int = 100
    ) -> PaginatedResults:
        """ (Docstring unchanged) """
        return await self._execute_with_retry(
            self.s2.search_author,
            query=query,
            fields=fields,
            limit=limit
        )


    async def get_recommended_papers(
        self,
        paper_id: str,
        fields: list = None,
        limit: int = 100,
        pool_from: Literal["recent", "all-cs"] = "recent"
    ) -> List[Paper]:
        """ (Docstring unchanged) """
        if pool_from not in ["recent", "all-cs"]:
             raise ValueError('The pool_from parameter must be either "recent" or "all-cs".')
             
        result = await self._execute_with_retry(
            self.s2.get_recommended_papers,
            paper_id=paper_id,
            fields=fields,
            limit=limit,
            pool_from=pool_from
        )
        # Recommendations API returns a list directly, not PaginatedResults
        return result if result is not None else []


    async def get_recommended_papers_from_lists(
        self,
        positive_paper_ids: List[str],
        negative_paper_ids: List[str] = None,
        fields: list = None,
        limit: int = 100
    ) -> List[Paper]:
        """ (Docstring unchanged) """
        result = await self._execute_with_retry(
            self.s2.get_recommended_papers_from_lists,
            positive_paper_ids=positive_paper_ids,
            negative_paper_ids=negative_paper_ids,
            fields=fields,
            limit=limit
        )
        # Recommendations API returns a list directly
        return result if result is not None else []

    async def get_autocomplete(self, query: str) -> List[Autocomplete]:
        """ (Docstring unchanged) """
        result = await self._execute_with_retry(
            self.s2.get_autocomplete,
            query=query
        )
        # Autocomplete API returns a list directly
        return result if result is not None else []

    def get_not_found_ids(self) -> List[str]:
        """
        Returns a list of IDs that resulted in ObjectNotFoundException
        during single-item lookups (get_paper, get_author).
        Note: Batch methods handle 'not found' internally via 'return_not_found'.
        """
        return list(self.not_found_ids)
