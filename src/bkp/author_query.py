import asyncio
from typing import List, Dict, Optional, Union, Tuple
from semanticscholar.Author import Author

from apis.s2_api import SemanticScholarKit
from apis.s2_data_process import process_author_data

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AuthorQuery:
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


    async def get_author_info(
            self,
            author_ids: List[str],
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
            with_abstract: Optional[bool] = True
    ) -> List[Dict]: # Return processed items instead of modifying state directly
        """Fetches and processes author information asynchronously."""
        if not author_ids:
            return []

        fields_of_study = fields_of_study if fields_of_study is not None else self.fields_of_study
        from_dt = from_dt if from_dt is not None else self.from_dt
        to_dt = to_dt if to_dt is not None else self.to_dt

        # search from Semantic Scholar for author data
        logging.info(f"Fetching info for {len(author_ids)} authors...")
        authors_metadata = await self.s2.get_authors(author_ids=author_ids) 
        authors_metadata = [item.raw_data for item in authors_metadata if isinstance(item, Author)]

        # check abstracts in author papers
        if with_abstract and len(authors_metadata) > 0:
            missing_abstract_s2ids = set()
            for info in authors_metadata:
                papers = info.get('papers', [])
                for paper in papers:
                    paper_id = paper.get('paperId')
                    abstract = paper.get('abstract')
                    if paper_id and abstract is None:
                        missing_abstract_s2ids.add(paper_id)

            if len(missing_abstract_s2ids) > 0:
                logging.info(f"Fetching abstracts for {len(missing_abstract_s2ids)} papers from authors (using controlled search).")
                paper_with_abstracts = await self.s2.get_papers(
                    paper_ids=list(missing_abstract_s2ids),
                    fields=['paperId', 'abstract'])
                ref_abstracts = {
                    item['paperId']: item['abstract']
                    for item in paper_with_abstracts
                    if item.get('paperId') is not None and item.get('abstract') is not None
                }

                if ref_abstracts:
                    logging.info(f"Found abstracts for {len(ref_abstracts)} papers.")
                    for info in authors_metadata:
                        papers = info.get('papers', [])
                        for paper in papers:
                            paper_id = paper.get('paperId')
                            if paper_id in ref_abstracts and paper.get('abstract') is None:
                                paper['abstract'] = ref_abstracts[paper_id]
                else:
                    logging.warning("Secondary search for author abstracts did not find any missing abstracts.")

        # process author information
        logging.info(f"Processing metadata for {len(authors_metadata)} authors.")
        processed_results = process_author_data(
            s2_authors=authors_metadata,
            from_dt=from_dt,
            to_dt=to_dt,
            fields_of_study=fields_of_study
        )

        # add not found authors
        found_author_ids = [author['authorId'] for author in authors_metadata if author.get('authorId') is not None]
        not_found_author_ids = set(author_ids) - set(found_author_ids)
        processed_results['not_found_info']['author'] = not_found_author_ids

        return processed_results
    

    async def get_paper_authors_info(
            self,
            paper_doi: List[str],
            limit: Optional[int] = 10,  # restrict number of authors
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
    ) -> List[Dict]: # Return processed items instead of modifying state directly
        """Fetches and processes author information asynchronously."""
        if not paper_doi:
            return []
        
        fields_of_study = fields_of_study if fields_of_study is not None else self.fields_of_study
        from_dt = from_dt if from_dt is not None else self.from_dt
        to_dt = to_dt if to_dt is not None else self.to_dt

        # search from Semantic Scholar for author data
        logging.info(f"Fetching author info for paper {paper_doi} ...")
        authors_metadata = await self.s2.get_paper_authors(paper_id=paper_doi, limit=limit) 
        authors_metadata = [item.raw_data for item in authors_metadata if isinstance(item, Author)]

        logging.info(f"Processing metadata for {len(authors_metadata)} authors.")
        processed_results = process_author_data(
            s2_authors=authors_metadata,
            from_dt=from_dt,
            to_dt=to_dt,
            fields_of_study=fields_of_study
        )

        return processed_results