import asyncio
from typing import List, Dict, Optional, Union, Tuple

from apis.s2_api import SemanticScholarKit
from apis.s2_data_process import process_citation_metadata

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CitationQuery:
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


    async def get_cited_papers(
            self,
            paper_doi: str,
            limit: Optional[int] = 100,
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
    ) -> List[Dict]: # Return processed items
        """Get papers cited by the paper asynchronously."""
        fields_of_study = fields_of_study if fields_of_study is not None else self.fields_of_study
        from_dt = from_dt if from_dt is not None else self.from_dt
        to_dt = to_dt if to_dt is not None else self.to_dt

        logging.info(f"Fetching papers cited by {paper_doi}...")
        s2_citedpaper_metadata = await self.s2.async_get_s2_cited_papers(paper_doi, limit=limit, with_abstract=True)

        logging.info(f"Processing {len(s2_citedpaper_metadata)} cited papers for {paper_doi}.")
        s2_citedpapermeta_json = process_citation_metadata(
            original_paper_doi=paper_doi,
            s2_citation_metadata=s2_citedpaper_metadata,
            citation_type='citedPaper', # Reference (paper cites this)
            from_dt=from_dt,
            to_dt=to_dt, 
            fields_of_study=fields_of_study
        )
        # Mark nodes
        for item in s2_citedpapermeta_json:
            if item['type'] == 'node' and item['labels'] == ['Paper']:
                item['properties']['from_reference'] = True
                item['properties']['is_complete'] = True
        return s2_citedpapermeta_json


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

        logging.info(f"Fetching papers citing {paper_doi}...")
        s2_citingpaper_metadata = await self.s2.async_get_s2_citing_papers(paper_doi, limit=limit, with_abstract=True)

        logging.info(f"Processing {len(s2_citingpaper_metadata)} citing papers for {paper_doi}.")
        s2_citingpapermetadata_json = process_citation_metadata(
            original_paper_doi=paper_doi,
            s2_citation_metadata=s2_citingpaper_metadata,
            citation_type='citingPaper', # Citation (this paper cites original)
            from_dt=from_dt,
            to_dt=to_dt,
            fields_of_study=fields_of_study
        )
        # Mark nodes
        for item in s2_citingpapermetadata_json:
            if item['type'] == 'node' and item['labels'] == ['Paper']:
                item['properties']['from_citation'] = True # Changed from 'from_citing'
                item['properties']['is_complete'] = True
        return s2_citingpapermetadata_json