import asyncio
from typing import List, Dict, Optional, Union, Tuple

from apis.s2_api import SemanticScholarKit
from apis.s2_data_process import process_citation_data

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CitationQuery:
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


    async def get_cited_papers(
            self,
            paper_doi: str,
            limit: Optional[int] = 100,
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
            with_abstract: Optional[bool] = True
    ) -> List[Dict]: # Return processed items
        """Get papers cited by the paper asynchronously."""
        fields_of_study = fields_of_study if fields_of_study is not None else self.fields_of_study
        from_dt = from_dt if from_dt is not None else self.from_dt
        to_dt = to_dt if to_dt is not None else self.to_dt

        # Search citation from Semantic Scholar
        logging.info(f"Fetching papers cited by {paper_doi}...")
        s2_cited_papers = await self.s2.get_paper_references(paper_id=paper_doi, limit=limit)

       # check abstracts in author papers
        if with_abstract and len(s2_cited_papers) > 0:
            missing_abstract_s2ids = set()
            for info in s2_cited_papers:
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
                    for info in s2_cited_papers:
                        papers = info.get('papers', [])
                        for paper in papers:
                            paper_id = paper.get('paperId')
                            if paper_id in ref_abstracts and paper.get('abstract') is None:
                                paper['abstract'] = ref_abstracts[paper_id]
                else:
                    logging.warning("Secondary search for author abstracts did not find any missing abstracts.")


        # Process citation data
        logging.info(f"Processing {len(s2_cited_papers)} cited papers for {paper_doi}.")
        processed_results = process_citation_data(
            original_paper_doi=paper_doi,
            s2_citations=s2_cited_papers,
            citation_type='citedPaper', # Reference (paper cites this)
            from_dt=from_dt,
            to_dt=to_dt, 
            fields_of_study=fields_of_study
        )

        # add not found dois
        if len(s2_cited_papers) > 0:   
            processed_results['not_found_info']['reference'] = set()
        else:
            processed_results['not_found_info']['reference'] = set({paper_doi})

        return processed_results


    async def get_citing_papers(
            self,
            paper_doi: str,
            limit: Optional[int] = 100,
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
    ) -> List[Dict]: # Return processed items
        """Retrieve papers that cite the paper asynchronously."""
        fields_of_study = fields_of_study if fields_of_study is not None else self.fields_of_study
        from_dt = from_dt if from_dt is not None else self.from_dt
        to_dt = to_dt if to_dt is not None else self.to_dt

        # Search citation from Semantic Scholar
        logging.info(f"Fetching papers citing {paper_doi}...")
        s2_citing_papers = await self.s2.async_get_s2_citing_papers(paper_doi, limit=limit, with_abstract=True)

        # Process citation data
        logging.info(f"Processing {len(s2_citing_papers)} citing papers for {paper_doi}.")
        processed_results = process_citation_data(
            original_paper_doi=paper_doi,
            s2_citations=s2_citing_papers,
            citation_type='citingPaper', # Citation (this paper cites original)
            from_dt=from_dt,
            to_dt=to_dt,
            fields_of_study=fields_of_study
        )

        # add not found dois
        if len(s2_citing_papers) > 0:   
            processed_results['not_found_info']['citing'] = set()
        else:
            processed_results['not_found_info']['citing'] = set({paper_doi})

        return processed_results