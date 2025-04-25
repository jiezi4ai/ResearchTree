import numpy as np
from typing import List, Dict, Optional, Union # Added Tuple

import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from models.embedding_models import gemini_embedding_async, semantic_similarity_matrix

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PaperSim:
    """
    A class for calculating paper similarities
    """
    def __init__(
        self,
        embed_api_key: Optional[str] = None,
        embed_model_name: Optional[str] = None,
    ):
        """Initialize parameters."""
        # embedding params
        self.embed_api_key = embed_api_key
        self.embed_model_name = embed_model_name

        # store paper abstraction embeddings
        self.abs_embed_ref = {}  # a k-v dict where key is paper doi and value is embedding of paper title and abstract

    async def get_abstract_embeds(
            self, 
            paper_nodes_json: Optional[List], 
            embed_api_key: Optional[str] = None,
            embed_model_name: Optional[str] = None,
            ):
        """Calculate semantic similarity asynchronously."""
        ids, texts = [], []
        for node in paper_nodes_json:
            node_id = node['id']
            title = node['properties'].get('title')
            abstract = node['properties'].get('abstract')
            if title is not None and abstract is not None:
                texts.append(f"TITLE: {title} \nABSTRACT: {abstract}")
                ids.append(node_id)

        if not texts:
            logging.warning("No paper titles and abstracts found for semantic similarity calculation.")
            return {}

        logging.info(f"Generating embeddings for {len(texts)} papers...")
        # Assume gemini_embedding_async handles its own batching/rate limits if necessary
        try:
             embeds = await gemini_embedding_async(embed_api_key, embed_model_name, texts, 10) # Pass batch size
        except Exception as e:
             logging.error(f"Embedding generation failed: {e}")
             return {}

        if embeds is None or (len(embeds) != len(texts)):
             logging.error(f"Embedding generation returned unexpected result. Expected {len(texts)} embeddings, got {len(embeds) if embeds else 0}.")
             return {}
        
        abs_embed_ref = {key: value for key, value in zip(ids, embeds)}    
        return abs_embed_ref
    

    def cal_paper_similarity(
            self,
            paper_nodes_json: List,
            paper_dois_1: List, 
            paper_dois_2: List,
            abs_embed_ref: List,
            similarity_threshold,
            ):
        semantic_similar_pool = []
        publish_dt_ref = {x['id']:x['properties'].get('publicationDate')
                        for x in paper_nodes_json if x['properties'].get('publicationDate') is not None}

        # Filter embeddings for each list
        embeds_ref_1 = {key: val for key, val in abs_embed_ref.items() if key in paper_dois_1}
        embeds_ref_2 = {key: val for key, val in abs_embed_ref.items() if key in paper_dois_2}

        # --- Start: Add Robustness Checks ---
        ids_1 = list(embeds_ref_1.keys())
        ids_2 = list(embeds_ref_2.keys())
        embed_values_1 = list(embeds_ref_1.values())
        embed_values_2 = list(embeds_ref_2.values())

        if not embed_values_1 or not embed_values_2:
            logging.warning(f"Cannot calculate similarity. "
                            f"Found {len(embed_values_1)} valid embeddings for list 1 (IDs: {ids_1}) and "
                            f"{len(embed_values_2)} for list 2 (IDs: {ids_2}). "
                            f"Check if input DOIs exist and have title/abstract.")
            return [] # Return empty list as no similarity can be calculated
        # --- End: Add Robustness Checks ---

        embeds_1 = np.array(embed_values_1)
        embeds_2 = np.array(embed_values_2)

        # Add logging for shapes *before* the calculation
        logging.info(f"Shape of embeds_1: {embeds_1.shape}")
        logging.info(f"Shape of embeds_2: {embeds_2.shape}")

        logging.info("Calculating similarity matrix...")
        try:
            # Assuming semantic_similarity_matrix handles potential normalization etc.
            # It likely computes cosine similarity: (embeds_1 @ embeds_2.T) / (norm(embeds_1) * norm(embeds_2))
            sim_matrix = semantic_similarity_matrix(embeds_1, embeds_2)
            sim_matrix = np.array(sim_matrix) # Ensure it's a numpy array if not already
        except Exception as e:
            # Log the shapes again in case of error
            logging.error(f"Similarity matrix calculation failed with embeds_1 shape {embeds_1.shape} and embeds_2 shape {embeds_2.shape}: {e}")
            return [] # Return empty list on failure

        logging.info("Processing similarity matrix to create relationships...")
        rows, cols = sim_matrix.shape
        added_pairs = set()

        # --- Small optimization/correction in loop ---
        if rows > 0 and cols > 0:
            # Ensure sim_matrix shape matches expectation: (len(ids_1), len(ids_2))
            if sim_matrix.shape != (len(ids_1), len(ids_2)):
                logging.error(f"Similarity matrix shape {sim_matrix.shape} does not match expected shape ({len(ids_1)}, {len(ids_2)})")
                return []

            for i in range(rows):      # Iterate through papers in list 1
                id_i = ids_1[i]
                publish_dt_i = publish_dt_ref.get(id_i)
                if publish_dt_i is None: # Skip if no publication date for comparison
                    continue

                for j in range(cols):  # Iterate through papers in list 2
                    id_j = ids_2[j]
                    # Avoid self-comparison if the lists could overlap and contain the same ID
                    if id_i == id_j:
                        continue

                    sim = sim_matrix[i, j]
                    if sim > similarity_threshold:
                        publish_dt_j = publish_dt_ref.get(id_j)
                        if publish_dt_j is None: # Skip if no publication date for comparison
                            continue

                        # Determine start/end based on publication date
                        if publish_dt_i <= publish_dt_j:
                            start_node_id = id_i
                            end_node_id = id_j
                        else:
                            start_node_id = id_j
                            end_node_id = id_i

                        # Create unique tuple for the pair (order matters for the relationship direction)
                        pair_tuple = (start_node_id, end_node_id)

                        if pair_tuple not in added_pairs:
                            edge = {
                                "type": "relationship",
                                "relationshipType": "SIMILAR_TO",
                                "startNodeId": start_node_id,
                                "endNodeId": end_node_id,
                                "properties": {
                                    'source': 'semantic similarity',
                                    'weight': round(float(sim), 4),
                                }
                            }
                            semantic_similar_pool.append(edge)
                            added_pairs.add(pair_tuple) # Store the directed pair
        else:
            logging.info("Similarity matrix is empty, no relationships to process.")

        return semantic_similar_pool
    

    async def cal_embed_and_similarity(
            self, 
            paper_nodes_json,
            paper_dois_1, 
            paper_dois_2,
            similarity_threshold,
            ):
        """calculate embeds and similarity"""
        paper_ids = [node['id'] for node in paper_nodes_json]
        input_paper_dois = list(set(paper_dois_1).union(paper_dois_2))

        status_1 = bool(set(paper_dois_1).intersection(set(paper_ids)))
        status_2 = bool(set(paper_dois_2).intersection(set(paper_ids)))
        if status_1 and status_2:
            # filter those not yet have emebedings
            paper_nodes_json_wo_embeds = [node for node in paper_nodes_json 
                                          if node['id'] in input_paper_dois
                                          and node['id'] not in self.abs_embed_ref]
            
            # calculate embeddings for those nodes
            if len(paper_nodes_json_wo_embeds) > 0:
                abs_embed_ref = await self.get_abstract_embeds(paper_nodes_json_wo_embeds, self.embed_api_key, self.embed_model_name) 
                # add embeddings
                self.abs_embed_ref.update(abs_embed_ref)

            semantic_similar_pool = self.cal_paper_similarity(
                paper_nodes_json, 
                paper_dois_1=paper_dois_1,
                paper_dois_2=paper_dois_2,
                abs_embed_ref=self.abs_embed_ref,
                similarity_threshold=similarity_threshold,
            )
            return semantic_similar_pool
        else:
            logging.error(f"The input {'' if status_1 else '1'} {'' if status_2 else '2'} paper dois have no intersection with current paper json.")
            return []