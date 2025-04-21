import math
import asyncio
from typing import List, Dict, Optional, Any, Callable, Union
import pyalex
from pyalex import Works, Authors, Venues, Concepts # Import necessary pyalex classes

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Set your email for the OpenAlex polite pool (optional but recommended)
# pyalex.config.email = "your_email@example.com"

MAX_CNT = 10000 # OpenAlex can often handle larger limits, adjust as needed
BATCH_SIZE = 50   # OpenAlex filter length limit - batching based on number of IDs in filter string
                  # Max URL length is ~8000 chars, 50 OpenAlex IDs (~2k chars) is safe.
DEFAULT_MAX_CONCURRENCY = 10 # Default concurrency limit
DEFAULT_SLEEP_INTERVAL = 0.1 # OpenAlex polite pool rate limit is 10 req/sec

# OpenAlex Max items per page
OPENALEX_MAX_PER_PAGE = 200

class OpenAlexKit:
    def __init__(
        self,
        email: str = None, # Added email parameter for polite pool
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        sleep_interval: float = DEFAULT_SLEEP_INTERVAL,
    ):
        """
        Initializes the OpenAlexKit.

        :param str email: (optional) Email for OpenAlex polite pool.
        :param int max_concurrency: Maximum number of concurrent API calls allowed.
        :param float sleep_interval: Seconds to sleep after each API call (non-blocking).
        """
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be a positive integer")
        if sleep_interval < 0:
            raise ValueError("sleep_interval must be non-negative")

        if email:
            pyalex.config.email = email
            logger.info(f"OpenAlex email set for polite pool: {email}")
        else:
            logger.warning("OpenAlex email not set. Using anonymous pool (lower rate limits).")

        self.batch_size = BATCH_SIZE
        self.max_cnt = MAX_CNT
        # --- Concurrency and Sleep Control ---
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.sleep_interval = sleep_interval
        logger.info(f"OpenAlexKit initialized with max_concurrency={max_concurrency}, sleep_interval={sleep_interval}s")
        # --- End Concurrency and Sleep Control ---

    async def _execute_sync_with_controls(self, sync_func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Executes a synchronous pyalex function in a thread pool, managed by a semaphore
        for concurrency control, and adds a non-blocking sleep afterwards.
        (Unchanged from original, works for any sync function)
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
                logger.error(f"Exception during controlled execution of {func_name}: {e}", exc_info=True)
                logger.debug(f"Sleeping for {self.sleep_interval}s after error in {func_name}.")
                await asyncio.sleep(self.sleep_interval)
                # Re-raise or return default based on expected output of sync_func
                # Returning empty list [] for functions expected to return lists
                if "search" in func_name or "get" in func_name:
                     return []
                raise e # Re-raise for other potential errors
            finally:
                logger.debug(f"Semaphore released for {func_name}.")


    # --- Synchronous Helper Methods (Adapted for pyalex) ---

    def _validate_openalex_ids(self, entity_type: str, ids: List[str]) -> List[str]:
        """Validates and filters IDs for OpenAlex format."""
        prefix_map = {"work": "W", "author": "A", "venue": "V", "concept": "C", "institution": "I"}
        prefix = prefix_map.get(entity_type.lower())
        if not prefix:
            raise ValueError(f"Unknown entity type for ID validation: {entity_type}")

        valid_ids = []
        for item_id in ids:
            # Basic check: is it a string and starts with the correct prefix?
            # More robust validation could involve regex or checking OpenAlex ID structure.
            if isinstance(item_id, str) and item_id.startswith(prefix):
                # Remove potential URL prefix
                if item_id.startswith("https://openalex.org/"):
                    valid_ids.append(item_id.split("/")[-1])
                else:
                    valid_ids.append(item_id)
            else:
                logger.warning(f"Invalid or non-{entity_type} ID format skipped: {item_id}")
        return valid_ids

    def _sync_get_works_by_ids(self, work_ids: List[str]) -> List[Dict]:
        """Fetches work details for a batch of OpenAlex Work IDs."""
        logger.info(f"_sync_get_works_by_ids: Thread started for batch ({len(work_ids)} IDs, first 5: {work_ids[:5]}...).")
        if not work_ids:
            return []
        try:
            # Construct the filter string with '|' for OR
            filter_query = "|".join(work_ids)
            # Use .get() which handles pagination internally for filter queries
            results = Works().filter(openalex_id=filter_query).get()
            logger.info(f"_sync_get_works_by_ids: API call successful for batch (first 5: {work_ids[:5]}...), returning {len(results)} items.")
            return results
        except Exception as e:
            logger.error(f"Error in _sync_get_works_by_ids for batch (first 5 IDs: {work_ids[:5]}...): {e}", exc_info=True)
            return []

    def _sync_get_authors_by_ids(self, author_ids: List[str]) -> List[Dict]:
        """Fetches author details for a batch of OpenAlex Author IDs."""
        logger.info(f"_sync_get_authors_by_ids: Thread started for batch ({len(author_ids)} IDs, first 5: {author_ids[:5]}...).")
        if not author_ids:
            return []
        try:
            filter_query = "|".join(author_ids)
            results = Authors().filter(openalex_id=filter_query).get()
            logger.info(f"_sync_get_authors_by_ids: API call successful for batch (first 5: {author_ids[:5]}...), returning {len(results)} items.")
            return results
        except Exception as e:
            logger.error(f"Error in _sync_get_authors_by_ids for batch (first 5 IDs: {author_ids[:5]}...): {e}", exc_info=True)
            return []

    def _sync_search_works_by_keywords(self, **kwargs) -> List[Dict]:
        """Searches OpenAlex works based on keywords and filters."""
        query = kwargs.get('query', None)
        limit = kwargs.get('limit', self.max_cnt)
        logger.info(f"_sync_search_works_by_keywords: Thread started for query '{str(query)[:50]}...' with limit {limit}.")

        try:
            works_query = Works()
            if query:
                works_query = works_query.search(query)

            # --- Apply Filters ---
            # Simple year filter
            if kwargs.get('year'):
                try:
                    works_query = works_query.filter(publication_year=int(kwargs['year']))
                except ValueError:
                    logger.warning(f"Invalid year format: {kwargs['year']}. Skipping year filter.")

            # Publication types (needs mapping from S2 terms if necessary)
            # Example: S2 'JournalArticle' maps to OA 'journal-article'
            oa_pub_types = kwargs.get('publication_types')
            if oa_pub_types:
                 # Assuming input types are already OpenAlex compatible or mapped beforehand
                 # Example: works_query = works_query.filter(type='journal-article')
                 # Handling multiple types with OR:
                 if isinstance(oa_pub_types, list):
                     type_filter = "|".join(oa_pub_types)
                     works_query = works_query.filter(type=type_filter)
                 elif isinstance(oa_pub_types, str):
                     works_query = works_query.filter(type=oa_pub_types)

            # Open Access PDF (using OA status flags)
            if kwargs.get('open_access_pdf'):
                # This checks for any OA version, adjust filter as needed (e.g., oa_status="gold")
                works_query = works_query.filter(has_oa_accepted_or_published_version=True) # Simplified check

            # Venue (Journal/Source) - Can filter by name or ID
            venues = kwargs.get('venue')
            if venues:
                if isinstance(venues, list):
                    venue_filter = "|".join(venues) # Assumes venue IDs or names suitable for OR filter
                    # Adjust filter key based on whether input is ID or name, e.g. primary_location={"source": {"id": venue_filter}}
                    # Using display_name for simplicity here:
                    works_query = works_query.filter(primary_location={"source": {"display_name": venue_filter}})
                elif isinstance(venues, str):
                     works_query = works_query.filter(primary_location={"source": {"display_name": venues}})


            # Fields of Study (Concepts) - Requires OpenAlex Concept IDs
            fields_of_study = kwargs.get('fields_of_study') # Assumed to be list of Concept IDs like 'C12345'
            if fields_of_study:
                 if isinstance(fields_of_study, list):
                     concept_filter = "|".join(self._validate_openalex_ids("concept", fields_of_study))
                     if concept_filter:
                         works_query = works_query.filter(concepts={"id": concept_filter})
                 elif isinstance(fields_of_study, str):
                      valid_concept = self._validate_openalex_ids("concept", [fields_of_study])
                      if valid_concept:
                          works_query = works_query.filter(concepts={"id": valid_concept[0]})

            # Publication Date or Year (Range or specific date)
            pub_date = kwargs.get('publication_date_or_year')
            if pub_date:
                # pyalex filter supports YYYY, YYYY-MM, YYYY-MM-DD and ranges >YYYY, <YYYY
                works_query = works_query.filter(publication_date=pub_date)

            # Minimum Citation Count
            min_citations = kwargs.get('min_citation_count')
            if min_citations is not None:
                 try:
                     count = int(min_citations)
                     if count >= 0:
                         works_query = works_query.filter(cited_by_count=f">{count}")
                 except ValueError:
                      logger.warning(f"Invalid min_citation_count: {min_citations}. Skipping filter.")

            # --- Sorting ---
            # Map S2 sort options ('relevance', 'citationCount', 'publicationDate') to OA fields
            sort_param = kwargs.get('sort')
            if sort_param:
                sort_field_map = {
                    'relevance': 'relevance_score',
                    'citationCount': 'cited_by_count',
                    'publicationDate': 'publication_date'
                }
                # Assumes format "field:asc" or "field:desc", default desc if not specified
                sort_parts = sort_param.split(':')
                field_s2 = sort_parts[0]
                direction = sort_parts[1] if len(sort_parts) > 1 else 'desc'

                if field_s2 in sort_field_map:
                    field_oa = sort_field_map[field_s2]
                    works_query = works_query.sort(**{field_oa: direction})
                else:
                    logger.warning(f"Unsupported sort field: {field_s2}. Using default OpenAlex relevance.")

            # --- Fetching Results with Pagination ---
            paper_metadata = []
            processed_count = 0
            # Use paginate to handle potential large result sets respecting the limit
            # Set per_page to OpenAlex max for efficiency
            page_size = min(limit, OPENALEX_MAX_PER_PAGE)
            if limit <= 0: page_size = OPENALEX_MAX_PER_PAGE # Fetch default if limit is 0 or less? Or return []?

            logger.info(f"Executing OpenAlex search query with limit={limit}, page_size={page_size}...")

            # Iterate through pages until the limit is reached
            for page in works_query.paginate(per_page=page_size, n_max=limit):
                 if not page: # Stop if a page is empty
                     break
                 # Add results from the page, ensuring not to exceed the overall limit
                 num_to_add = min(len(page), limit - processed_count)
                 paper_metadata.extend(page[:num_to_add])
                 processed_count += num_to_add
                 logger.debug(f"Fetched page, total processed: {processed_count}/{limit}")
                 if processed_count >= limit:
                     break # Exit loop once limit is reached

            logger.info(f"_sync_search_works_by_keywords: API call successful for query '{str(query)[:50]}...', returning {len(paper_metadata)} items.")
            return paper_metadata

        except Exception as e:
            logger.error(f"Error in _sync_search_works_by_keywords for query '{str(query)[:50]}...': {e}", exc_info=True)
            return []

    def _sync_get_work_references(self, work_id: str, limit: int) -> List[Dict]:
        """Gets works cited by the given work (its references)."""
        logger.info(f"_sync_get_work_references: Thread started for work {work_id} with limit {limit}.")
        if not work_id: return []
        try:
            # 1. Get the work itself to find its references
            work_data = Works()[work_id].get()
            if not work_data or 'referenced_works' not in work_data:
                logger.warning(f"Work {work_id} not found or has no referenced_works field.")
                return []

            # 2. Extract referenced work IDs (these are full OpenAlex URLs, need to extract IDs)
            referenced_urls = work_data.get('referenced_works', [])
            if not referenced_urls:
                 logger.info(f"Work {work_id} has no references listed.")
                 return []

            # Extract IDs from URLs and apply limit
            referenced_ids = [url.split('/')[-1] for url in referenced_urls if url]
            referenced_ids_limited = self._validate_openalex_ids("work", referenced_ids[:limit])


            # 3. Fetch the details of the referenced works in batches if necessary
            # Since _sync_get_works_by_ids handles batching internally via filter, we can pass all limited IDs
            if not referenced_ids_limited:
                logger.info(f"No valid referenced work IDs found for {work_id} after filtering.")
                return []

            logger.info(f"Fetching details for {len(referenced_ids_limited)} referenced works for {work_id}.")
            # This call itself doesn't need the semaphore wrapper again, as it's part of the sync task
            referenced_works_data = self._sync_get_works_by_ids(referenced_ids_limited)

            logger.info(f"_sync_get_work_references: API call successful for work {work_id}, returning {len(referenced_works_data)} referenced items.")
            return referenced_works_data

        except Exception as e:
            logger.error(f"Error in _sync_get_work_references for {work_id}: {e}", exc_info=True)
            return []

    def _sync_get_work_citations(self, work_id: str, limit: int) -> List[Dict]:
        """Gets works that cite the given work."""
        logger.info(f"_sync_get_work_citations: Thread started for work {work_id} with limit {limit}.")
        if not work_id: return []
        try:
            # Use the 'cites' filter in OpenAlex
            citing_works_query = Works().filter(cites=work_id)

            # Fetch results with pagination, respecting the limit
            citations_metadata = []
            processed_count = 0
            page_size = min(limit, OPENALEX_MAX_PER_PAGE)
            if limit <= 0: page_size = OPENALEX_MAX_PER_PAGE

            logger.info(f"Executing OpenAlex citations query for {work_id} with limit={limit}, page_size={page_size}...")

            for page in citing_works_query.paginate(per_page=page_size, n_max=limit):
                if not page:
                    break
                num_to_add = min(len(page), limit - processed_count)
                citations_metadata.extend(page[:num_to_add])
                processed_count += num_to_add
                logger.debug(f"Fetched citations page, total processed: {processed_count}/{limit}")
                if processed_count >= limit:
                    break

            logger.info(f"_sync_get_work_citations: API call successful for work {work_id}, returning {len(citations_metadata)} citing items.")
            return citations_metadata

        except Exception as e:
            logger.error(f"Error in _sync_get_work_citations for {work_id}: {e}", exc_info=True)
            return []

    # --- Asynchronous Public Methods (Interface Unchanged, Logic Updated for OpenAlex) ---
    async def async_search_paper_by_ids(
        self,
        id_list: List[str] # Expecting OpenAlex Work IDs (e.g., W12345)
    ) -> List[Dict]:
        """Search paper by OpenAlex Work IDs asynchronously."""
        # Validate IDs first
        valid_id_list = self._validate_openalex_ids("work", id_list)
        id_cnt = len(valid_id_list)
        paper_metadata = []

        if id_cnt > 0:
            # Batching based on number of IDs per filter query
            batch_size = self.batch_size
            batch_cnt = math.ceil(id_cnt / batch_size)
            batches = [valid_id_list[i * batch_size:(i + 1) * batch_size] for i in range(batch_cnt)]

            tasks = []
            logger.info(f"async_search_paper_by_ids: Creating {len(batches)} tasks for {id_cnt} valid OpenAlex Work IDs.")
            for batch in batches:
                # Use the controlled execution wrapper for each batch task
                tasks.append(self._execute_sync_with_controls(self._sync_get_works_by_ids, batch))

            logger.info(f"async_search_paper_by_ids: Gathering {len(tasks)} tasks...")
            batch_results_list = await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(f"async_search_paper_by_ids: Gather complete. Processing results.")

            # Aggregate results
            for result in batch_results_list:
                if isinstance(result, Exception):
                    logger.error(f"A batch task for async_search_paper_by_ids failed: {result}")
                elif isinstance(result, list):
                    paper_metadata.extend(result)
                else:
                    logger.warning(f"Unexpected result type {type(result)} from paper batch task: {result}")
        else:
            logger.warning("async_search_paper_by_ids: No valid OpenAlex Work IDs provided.")

        return paper_metadata

    async def async_search_author_by_ids(
        self,
        author_ids: List[str], # Expecting OpenAlex Author IDs (e.g., A12345)
        with_abstract: Optional[bool] = False # Note: OpenAlex authors don't directly list abstracts
    ) -> List[Dict]:
        """Search author by OpenAlex Author IDs asynchronously."""
        valid_id_list = self._validate_openalex_ids("author", author_ids)
        id_cnt = len(valid_id_list)
        author_metadata = []

        if id_cnt > 0:
            batch_size = self.batch_size
            batch_cnt = math.ceil(id_cnt / batch_size)
            batches = [valid_id_list[i * batch_size:(i + 1) * batch_size] for i in range(batch_cnt)]
            logger.info(f"async_search_author_by_ids: Fetching {id_cnt} authors by ID in {batch_cnt} batches.")

            tasks = []
            for batch in batches:
                 tasks.append(self._execute_sync_with_controls(self._sync_get_authors_by_ids, batch))

            logger.info(f"async_search_author_by_ids: Gathering {len(tasks)} tasks...")
            batch_results_list = await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(f"async_search_author_by_ids: Gather complete. Processing results.")

            for result in batch_results_list:
                if isinstance(result, Exception):
                    logger.error(f"A batch task for async_search_author_by_ids failed: {result}")
                elif isinstance(result, list):
                    author_metadata.extend(result)
                else:
                     logger.warning(f"Unexpected result type {type(result)} from author batch task: {result}")
        else:
             logger.warning("async_search_author_by_ids: No valid OpenAlex Author IDs provided.")


        # --- Handle with_abstract ---
        # OpenAlex Author objects don't contain paper abstracts directly.
        # This section *could* be adapted to fetch the author's works and then fetch abstracts for those works,
        # but it significantly changes the scope and performance.
        # For now, we'll log a warning if with_abstract is True, as it's not directly applicable here.
        if with_abstract:
            logger.warning("`with_abstract=True` is not directly supported for `async_search_author_by_ids` with OpenAlex. "
                           "Author data does not contain paper abstracts. Fetch author's works separately if needed.")
            # If implementation is desired:
            # 1. Extract work IDs associated with each author (e.g., from author['counts_by_year']) - complex parsing needed
            # 2. Collect all unique work IDs.
            # 3. Call `async_search_paper_by_ids` for these work IDs.
            # 4. Map abstracts back (difficult as author object doesn't list papers directly).

        return author_metadata

    async def async_search_paper_by_keywords(
        self,
        query: str,
        year: str = None,
        publication_types: list = None, # Expect OpenAlex type strings e.g. ['journal-article', 'book-chapter']
        open_access_pdf: bool = None, # Simplified OA check
        venue: list = None, # Expect Venue display names or OpenAlex Venue IDs
        fields_of_study: list = None, # Expect OpenAlex Concept IDs
        publication_date_or_year: str = None, # e.g. "2020", ">2019", "2020-01-01"
        min_citation_count: int = None,
        limit: int = 100,
        bulk: bool = False, # 'bulk' not directly used by pyalex, limit handles amount.
        sort: str = None, # e.g. "citationCount:desc", "publicationDate:asc", "relevance:desc"
        match_title: bool = False # Use search:'query' for general search, specific title search needs field query
    ) -> List[Dict]:
        """Search for papers (works) by keyword and filters asynchronously using OpenAlex."""

        # Prepare arguments for the synchronous search function
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
            # bulk is implicitly handled by pagination up to limit
            "sort": sort,
            # match_title could be implemented by modifying the query string if needed, e.g. query=f'title:{query}'
            # but pyalex search() might handle this implicitly to some extent. Keeping it simple for now.
        }
        # Remove None values to avoid passing them to filters unnecessarily
        search_kwargs = {k: v for k, v in search_kwargs.items() if v is not None}

        logger.info(f"async_search_paper_by_keywords: Searching papers by keyword: '{query[:50]}...' with effective limit {search_kwargs.get('limit')}.")
        try:
             # Use the controlled execution wrapper
            paper_metadata = await self._execute_sync_with_controls(self._sync_search_works_by_keywords, **search_kwargs)
        except Exception as e:
             logger.error(f"async_search_paper_by_keywords: Failed for query '{query[:50]}...': {e}")
             paper_metadata = [] # Return empty list on failure

        return paper_metadata

    async def async_get_s2_cited_papers( # Renamed to reflect S2 origin, maps to OpenAlex References
        self,
        paper_id: str, # Expecting a single OpenAlex Work ID
        limit: int = 100,
        with_abstract: Optional[bool] = False
    ) -> List[Dict]:
        """Get papers cited by this paper (references) asynchronously using OpenAlex."""
        valid_paper_id = self._validate_openalex_ids("work", [paper_id])
        if not valid_paper_id:
             logger.error(f"async_get_s2_cited_papers: Invalid OpenAlex Work ID provided: {paper_id}")
             return []
        work_id = valid_paper_id[0] # Use the validated ID

        max_limit = min(limit, self.max_cnt)
        logger.info(f"async_get_s2_cited_papers: Fetching references for paper {work_id} with effective limit {max_limit}.")

        refs_metadata = []
        try:
            # Use the controlled execution wrapper for the sync reference fetching function
            refs_metadata = await self._execute_sync_with_controls(self._sync_get_work_references, work_id, max_limit)
        except Exception as e:
             logger.error(f"async_get_s2_cited_papers: Failed for paper {work_id}: {e}")
             refs_metadata = []

        # --- Handle with_abstract ---
        # Check if abstracts are missing and fetch them if requested
        if with_abstract and refs_metadata:
            papers_missing_abstracts_ids = set()
            for paper in refs_metadata:
                # OpenAlex stores abstract in 'abstract_inverted_index' (aii).
                # pyalex reconstructs 'abstract' field if aii is present. Check for None 'abstract'.
                p_id = paper.get('id')
                abstract = paper.get('abstract')
                if p_id and abstract is None:
                    # Only add valid OpenAlex IDs
                    if isinstance(p_id, str) and p_id.startswith("W"):
                         papers_missing_abstracts_ids.add(p_id)

            if papers_missing_abstracts_ids:
                logger.info(f"Fetching potentially missing abstracts for {len(papers_missing_abstracts_ids)} cited papers (references).")
                # This call uses the OpenAlex implementation internally now
                paper_metadata_with_abstracts = await self.async_search_paper_by_ids(id_list=list(papers_missing_abstracts_ids))

                ref_abstracts = {
                    item['id']: item['abstract']
                    for item in paper_metadata_with_abstracts
                    if item.get('id') and item.get('abstract') # Check if abstract was found
                }

                if ref_abstracts:
                    logger.info(f"Found abstracts for {len(ref_abstracts)} cited papers.")
                    # Add abstracts back to the original list
                    for paper in refs_metadata:
                        p_id = paper.get('id')
                        if p_id in ref_abstracts and paper.get('abstract') is None:
                            paper['abstract'] = ref_abstracts[p_id]
                else:
                    logger.info("Secondary search did not find any missing abstracts for cited papers (they might not be available in OpenAlex).")

        # S2 format was [{'citedPaper': {...}}, ...]. OpenAlex returns [{...}, ...].
        # Returning the OpenAlex format directly. If S2 structure is strictly needed, wrap here.
        return refs_metadata

    async def async_get_s2_citing_papers( # Renamed to reflect S2 origin, maps to OpenAlex Citations
        self,
        paper_id: str, # Expecting a single OpenAlex Work ID
        limit: int = 100,
        with_abstract: Optional[bool] = False
    ) -> List[Dict]:
        """Get papers citing this paper asynchronously using OpenAlex."""
        valid_paper_id = self._validate_openalex_ids("work", [paper_id])
        if not valid_paper_id:
             logger.error(f"async_get_s2_citing_papers: Invalid OpenAlex Work ID provided: {paper_id}")
             return []
        work_id = valid_paper_id[0]

        max_limit = min(limit, self.max_cnt)
        logger.info(f"async_get_s2_citing_papers: Fetching citations for paper {work_id} with effective limit {max_limit}.")

        citedby_metadata = []
        try:
             # Use the controlled execution wrapper for the sync citation fetching function
            citedby_metadata = await self._execute_sync_with_controls(self._sync_get_work_citations, work_id, max_limit)
        except Exception as e:
            logger.error(f"async_get_s2_citing_papers: Failed for paper {work_id}: {e}")
            citedby_metadata = []

        # --- Handle with_abstract ---
        if with_abstract and citedby_metadata:
            papers_missing_abstracts_ids = set()
            for paper in citedby_metadata:
                p_id = paper.get('id')
                abstract = paper.get('abstract') # Check reconstructed abstract
                if p_id and abstract is None:
                    if isinstance(p_id, str) and p_id.startswith("W"):
                        papers_missing_abstracts_ids.add(p_id)

            if papers_missing_abstracts_ids:
                logger.info(f"Fetching potentially missing abstracts for {len(papers_missing_abstracts_ids)} citing papers.")
                paper_metadata_with_abstracts = await self.async_search_paper_by_ids(id_list=list(papers_missing_abstracts_ids))
                ref_abstracts = {
                    item['id']: item['abstract']
                    for item in paper_metadata_with_abstracts
                    if item.get('id') and item.get('abstract')
                }

                if ref_abstracts:
                    logger.info(f"Found abstracts for {len(ref_abstracts)} citing papers.")
                    for paper in citedby_metadata:
                        p_id = paper.get('id')
                        if p_id in ref_abstracts and paper.get('abstract') is None:
                            paper['abstract'] = ref_abstracts[p_id]
                else:
                    logger.info("Secondary search did not find any missing abstracts for citing papers.")

        # S2 format was [{'citingPaper': {...}}, ...]. OpenAlex returns [{...}, ...].
        # Returning the OpenAlex format directly.
        return citedby_metadata

    async def async_get_s2_recommended_papers(
        self,
        positive_paper_ids: List[str], # Expecting OpenAlex Work IDs
        negative_paper_ids: List[str] = None, # Expecting OpenAlex Work IDs
        limit: int = 100,
        with_abstract: Optional[bool] = False # Keep param for signature consistency
    ) -> List[Dict]:
        """
        Get recommended papers based on positive/negative examples.
        NOTE: OpenAlex does not have a direct equivalent to Semantic Scholar's
              recommendation engine based on positive/negative ID lists.
              This function will return an empty list.
        """
        logger.warning("OpenAlex does not support recommendations based on positive/negative paper ID lists like Semantic Scholar.")
        logger.warning("async_get_s2_recommended_papers will return an empty list.")

        # Validate inputs for logging clarity, though they won't be used for API calls
        valid_pos_ids = self._validate_openalex_ids("work", positive_paper_ids)
        valid_neg_ids = []
        if negative_paper_ids:
             valid_neg_ids = self._validate_openalex_ids("work", negative_paper_ids)

        logger.info(f"async_get_s2_recommended_papers called with {len(valid_pos_ids)} positive and {len(valid_neg_ids)} negative IDs. Limit: {limit}. Returning [].")

        # Return empty list as per the limitation
        return []