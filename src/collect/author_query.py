import asyncio
from typing import List, Dict, Optional, Union, Tuple

from apis.s2_api import SemanticScholarKit
from apis.s2_data_process import process_author_metadata

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AuthorQuery:
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


    async def get_author_info(
            self,
            author_ids: List[str],
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
    ) -> List[Dict]: # Return processed items instead of modifying state directly
        """Fetches and processes author information asynchronously."""
        if not author_ids:
            return []

        fields_of_study = fields_of_study if fields_of_study is not None else self.fields_of_study
        from_dt = from_dt if from_dt is not None else self.from_dt
        to_dt = to_dt if to_dt is not None else self.to_dt

        logging.info(f"Fetching info for {len(author_ids)} authors...")
        authors_info = await self.s2.async_search_author_by_ids(author_ids=author_ids, with_abstract=True) # Assuming fields apply here too

        logging.info(f"Processing metadata for {len(authors_info)} authors.")
        s2_author_meta_json = process_author_metadata(
            s2_author_metadata=authors_info,
            from_dt=from_dt,
            to_dt=to_dt,
            fields_of_study=fields_of_study
        )

        # Mark nodes appropriately before returning
        for item in s2_author_meta_json:
            if item['type'] == 'node':
                if item['labels'] == ['Paper']:
                    item['properties']['from_same_author'] = True
                    item['properties']['is_complete'] = True 
                elif item['labels'] == ['Author'] and item['id'] in author_ids:
                    item['properties']['is_complete'] = True
        return s2_author_meta_json
    

