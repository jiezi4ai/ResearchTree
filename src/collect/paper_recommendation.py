import asyncio
from typing import List, Dict, Optional, Union, Tuple

from apis.s2_api import SemanticScholarKit
from apis.s2_data_process import reset_and_filter_paper, process_paper_metadata

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PaperRecommendation:
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


    async def get_recommend_papers(
            self,
            paper_dois: List[str], # Expecting list now based on gather usage
            limit: Optional[int] = 100,
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
    ) -> List[Dict]: # Return processed items
        """Retrieve recommended papers asynchronously."""
        if not paper_dois: return []

        fields_of_study = fields_of_study if fields_of_study is not None else self.fields_of_study
        from_dt = from_dt if from_dt is not None else self.from_dt
        to_dt = to_dt if to_dt is not None else self.to_dt

        logging.info(f"Fetching recommendations based on {len(paper_dois)} papers...")
        s2_recommended_metadata = await self.s2.async_get_s2_recommended_papers(positive_paper_ids=paper_dois, limit=limit, with_abstract=True)

        logging.info(f"Processing {len(s2_recommended_metadata)} recommended papers.")
        # reset id and filter papers 
        filtered_s2_paper_metadata = reset_and_filter_paper(
            s2_paper_metadata=s2_recommended_metadata,
            from_dt=from_dt, 
            to_dt=to_dt, 
            fields_of_study=fields_of_study)
        s2_recpapermetadata_json = process_paper_metadata(filtered_s2_paper_metadata)

        # Mark nodes
        for item in s2_recpapermetadata_json:
            if item['type'] == 'node' and item['labels'] == ['Paper']:
                item['properties']['from_recommended'] = True
                item['properties']['is_complete'] = True
        return s2_recpapermetadata_json