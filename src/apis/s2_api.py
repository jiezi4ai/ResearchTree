import math
import asyncio
from typing import List, Dict, Optional, Any
from semanticscholar import SemanticScholar, Paper, Author

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAX_CNT = 100  # set limit max
BATCH_SIZE = 100  # S2 API batch size restriction

class SemanticScholarKit:
    def __init__(
            self, 
            ss_api_key: str = None, 
            ss_api_url: str = None,
            ):
        """
        :param str ss_api_key: (optional) private API key.
        :param str ss_api_url: (optional) custom API url.
        """
        self.scholar = SemanticScholar(api_key=ss_api_key, api_url=ss_api_url)
        self.batch_size = BATCH_SIZE
        self.max_cnt = MAX_CNT

    def _sync_get_papers(self, paper_ids: List[str]) -> List[Dict]:
        logger.info(f"_sync_get_papers: Thread started for batch (first 5: {paper_ids[:5]}...).") # Added Log
        try:
            batch_results: List[Paper] = self.scholar.get_papers(paper_ids=paper_ids)
            processed_results = [item._data for item in batch_results if hasattr(item, '_data')]
            logger.info(f"_sync_get_papers: API call successful for batch (first 5: {paper_ids[:5]}...), returning {len(processed_results)} items.") # Added Log
            return processed_results
        except Exception as e:
            logger.error(f"Error in _sync_get_papers for batch (first 5 IDs: {paper_ids[:5]}...): {e}", exc_info=True) # Added exc_info
            return [] # Still returning empty list on error

    async def async_search_paper_by_ids(
        self,
        id_list: List[str]
    ) -> List[Dict]:
        """Search paper by id asynchronously using asyncio.to_thread."""
        valid_id_list = [x for x in id_list if x and isinstance(x, str)]
        id_cnt = len(valid_id_list)
        paper_metadata = []

        if id_cnt > 0:
            # Batching logic remains
            batch_size = self.batch_size 
            batch_cnt = math.ceil(id_cnt / batch_size)
            batches = [valid_id_list[i * batch_size:(i + 1) * batch_size] for i in range(batch_cnt)]
  
            tasks = []
            for batch in batches:
                tasks.append(asyncio.to_thread(self._sync_get_papers, batch))

            logger.info(f"search_paper_by_ids: Gathering {len(tasks)} tasks...")
            try:
                batch_results_list = await asyncio.gather(*tasks, return_exceptions=True)
                logger.info(f"search_paper_by_ids: Gather complete. Processing results.")
                # Aggregate results
                for result in batch_results_list:
                    if isinstance(result, Exception):
                        logger.error(f"A batch task for search_paper_by_ids failed: {result}")
                        # Decide how to handle partial failures (e.g., log, skip, raise)
                    elif isinstance(result, list):
                        paper_metadata.extend(result)
                    else:
                        logger.warning(f"Unexpected result type from batch task: {type(result)}")
            except Exception as e:
                logger.error(f"search_paper_by_ids: Exception during asyncio.gather: {e}", exc_info=True) # Added Log
                paper_metadata = []

        return paper_metadata

    def _sync_get_authors(self, author_ids: List[str]) -> List[Dict]:
        """Synchronous helper for the actual API call."""
        try:
            batch_results: List[Author] = self.scholar.get_authors(author_ids=author_ids)
            return [item._data for item in batch_results if hasattr(item, '_data')]
        except Exception as e:
            logger.error(f"Error in _sync_get_authors for batch (first 5 IDs: {author_ids[:5]}...): {e}")
            return [] # Or raise e

    async def async_search_author_by_ids(
        self,
        author_ids: List[str],
        with_abstract: Optional[bool] = False
    ) -> List[Dict]:
        """Search author by ids asynchronously."""
        valid_id_list = [x for x in author_ids if x and isinstance(x, str)]
        id_cnt = len(valid_id_list)
        author_metadata = []

        if id_cnt > 0:
            batch_size = self.batch_size
            batch_cnt = math.ceil(id_cnt / batch_size)
            batches = [valid_id_list[i * batch_size:(i + 1) * batch_size] for i in range(batch_cnt)]
            logger.info(f"Fetching {id_cnt} authors by ID in {batch_cnt} batches.")

            tasks = []
            for batch in batches:
                # Pass fields to the sync helper
                tasks.append(asyncio.to_thread(self._sync_get_authors, batch))

            batch_results_list = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results_list:
                if isinstance(result, Exception):
                    logger.error(f"A batch task for search_author_by_ids failed: {result}")
                elif isinstance(result, list):
                    author_metadata.extend(result)
                else:
                    logger.warning(f"Unexpected result type from author batch task: {type(result)}")

        # --- Handle with_abstract asynchronously ---
        if with_abstract and author_metadata:
            papers_missing_abstract_ids = set() # Use set for efficiency
            for info in author_metadata:
                papers = info.get('papers', [])
                for paper in papers:
                    paper_id = paper.get('paperId')
                    abstract = paper.get('abstract')
                    if paper_id and abstract is None:
                        papers_missing_abstract_ids.add(paper_id)

            if papers_missing_abstract_ids:
                logger.info(f"Fetching abstracts for {len(papers_missing_abstract_ids)} papers from authors.")
                paper_metadata_with_abstracts = await self.async_search_paper_by_ids(
                    id_list=list(papers_missing_abstract_ids)
                )
                ref_abstracts = {
                    item['paperId']: item['abstract']
                    for item in paper_metadata_with_abstracts
                    if item.get('paperId') and item.get('abstract')
                }

                # Update original author_metadata in place
                if ref_abstracts:
                    logger.info(f"Found abstracts for {len(ref_abstracts)} papers.")
                    for info in author_metadata:
                        papers = info.get('papers', [])
                        for paper in papers:
                            paper_id = paper.get('paperId')
                            if paper_id in ref_abstracts and paper.get('abstract') is None:
                                paper['abstract'] = ref_abstracts[paper_id]
                else:
                    logger.warning("Secondary search did not find any missing abstracts.")

        return author_metadata

    def _sync_search_paper_by_keywords(self, **kwargs) -> List[Dict]:
        """Synchronous helper for keyword search."""
        try:
            results = self.scholar.search_paper(**kwargs)
            limit = kwargs.get('limit', self.max_cnt) # Get limit from args
            paper_metadata = []
            if results and hasattr(results, 'total') and results.total > 0:
                # Iterate through results safely, respecting limit
                count = 0
                for item in results:
                    if count >= limit:
                        break
                    if hasattr(item, '_data'):
                        paper_metadata.append(item._data)
                    count += 1
            return paper_metadata
        except Exception as e:
            logger.error(f"Error in _sync_search_paper_by_keywords for query '{kwargs.get('query')}': {e}")
            return []

    async def async_search_paper_by_keywords(
        self,
        query: str,
        year: str = None,
        publication_types: list = None,
        open_access_pdf: bool = None,
        venue: list = None,
        fields_of_study: list = None, # Note: This might overlap/conflict with 'fields' parameter - check S2 API docs
        publication_date_or_year: str = None,
        min_citation_count: int = None,
        limit: int = 100,
        bulk: bool = False,
        sort: str = None,
        match_title: bool = False
    ) -> List[Dict]:
        """Search for papers by keyword asynchronously."""
        # Pass all keyword arguments to the sync helper
        search_kwargs = {
            "query": query,
            "year": year,
            "publication_types": publication_types,
            "open_access_pdf": open_access_pdf,
            "venue": venue,
            "fields_of_study": fields_of_study, 
            "publication_date_or_year": publication_date_or_year,
            "min_citation_count": min_citation_count,
            "limit": min(limit, self.max_cnt), # Respect limit, ensure <= 100 unless bulk
            "bulk": bulk,
            "sort": sort,
            "match_title": match_title,
        }
        # Remove None values as the underlying library might not handle them
        search_kwargs = {k: v for k, v in search_kwargs.items() if v is not None}

        logger.info(f"Searching papers by keyword: '{query[:50]}...' with limit {search_kwargs.get('limit')}.")
        paper_metadata = await asyncio.to_thread(self._sync_search_paper_by_keywords, **search_kwargs)
        return paper_metadata


    def _sync_get_paper_references(self, paper_id: str, limit: int) -> List[Dict]:
        """Synchronous helper for getting references."""
        try:
            # Pass fields to the underlying library call
            results = self.scholar.get_paper_references(paper_id, limit=limit)
            return [item._data for item in results if hasattr(item, '_data')]
        except Exception as e:
            logger.error(f"Error in _sync_get_paper_references for {paper_id}: {e}")
            return []

    async def async_get_s2_cited_papers( # Corresponds to get_paper_references
        self,
        paper_id: str,
        limit: int = 100,
        with_abstract: Optional[bool] = False
    ) -> List[Dict]:
        """Get papers cited by this paper (references) asynchronously."""
        max_limit = min(limit, self.max_cnt)
        logger.info(f"Fetching references for paper {paper_id} with limit {max_limit}.")

        # Get initial reference data
        refs_metadata = await asyncio.to_thread(self._sync_get_paper_references, paper_id, max_limit)

        # --- Handle with_abstract asynchronously ---
        if with_abstract and refs_metadata:
            papers_missing_abstract_ids = set()
            for info in refs_metadata:
                paper = info.get('citedPaper') # Note: key is 'citedPaper'
                if isinstance(paper, dict):
                    p_id = paper.get('paperId')
                    abstract = paper.get('abstract')
                    if p_id and abstract is None:
                        papers_missing_abstract_ids.add(p_id)

            if papers_missing_abstract_ids:
                logger.info(f"Fetching abstracts for {len(papers_missing_abstract_ids)} cited papers (references).")
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


    def _sync_get_paper_citations(self, paper_id: str, limit: int) -> List[Dict]:
        """Synchronous helper for getting citations."""
        try:
            # Pass fields to the underlying library call
            results = self.scholar.get_paper_citations(paper_id, limit=limit)
            return [item._data for item in results if hasattr(item, '_data')]
        except Exception as e:
            logger.error(f"Error in _sync_get_paper_citations for {paper_id}: {e}")
            return []

    async def async_get_s2_citing_papers( # Corresponds to get_paper_citations
        self,
        paper_id: str,
        limit: int = 100,
        with_abstract: Optional[bool] = False
    ) -> List[Dict]:
        """Get papers citing this paper asynchronously."""
        max_limit = min(limit, self.max_cnt) # Respect API limit for citations
        logger.info(f"Fetching citations for paper {paper_id} with limit {max_limit}.")

        # Get initial citation data
        citedby_metadata = await asyncio.to_thread(self._sync_get_paper_citations, paper_id, max_limit)

        # --- Handle with_abstract asynchronously ---
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

    def _sync_get_recommended_papers(self, positive_paper_ids: List[str], negative_paper_ids: Optional[List[str]], limit: int) -> List[Dict]:
        """Synchronous helper for getting recommendations."""
        try:
            # Pass fields to the underlying library call
            results = self.scholar.get_recommended_papers_from_lists(
                positive_paper_ids=positive_paper_ids,
                negative_paper_ids=negative_paper_ids,
                limit=limit
            )
            return [item._data for item in results if hasattr(item, '_data')]
        except Exception as e:
            logger.error(f"Error in _sync_get_recommended_papers regarding {positive_paper_ids[:5]}...: {e}")
            return []

    async def async_get_s2_recommended_papers(
        self,
        positive_paper_ids: List[str],
        negative_paper_ids: List[str] = None,
        limit: int = 100,
        with_abstract: Optional[bool] = False
    ) -> List[Dict]:
        """Get recommended papers asynchronously."""
        max_limit = min(limit, self.max_cnt) # Respect API limit for recommendations
        logger.info(f"Fetching recommendations based on {len(positive_paper_ids)} positive IDs with limit {max_limit}.")

        rec_metadata = await asyncio.to_thread(self._sync_get_recommended_papers, positive_paper_ids, negative_paper_ids, max_limit)

        # --- Handle with_abstract asynchronously ---
        if with_abstract and rec_metadata:
            papers_missing_abstract_ids = set()
            for paper in rec_metadata: # Recommended papers are directly in the list
                paper_id = paper.get('paperId')
                abstract = paper.get('abstract')
                if paper_id and abstract is None:
                    papers_missing_abstract_ids.add(paper_id)

            if papers_missing_abstract_ids:
                logger.info(f"Fetching abstracts for {len(papers_missing_abstract_ids)} recommended papers.")
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