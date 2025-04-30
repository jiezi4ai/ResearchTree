import numpy as np
from typing import List, Dict, Optional, Union # Added Tuple

import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from models.embedding_models import gemini_embedding_async, semantic_similarity_matrix
from collect.paper_data_process import process_p2p_sim_data, process_p2t_sim_data

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
        self.abs_embed_ref = {}  # a k-v dict where key is paper id and value is embedding of paper title and abstract
        self.topic_embed_ref = {}  # a k-v dict where key is topic id and value is embedding of paper title and abstract


    async def get_topic_embeds(
            self,
            topic_nodes_json: Optional[List], 
            embed_api_key: Optional[str] = None,
            embed_model_name: Optional[str] = None, 
            ):
        """Calculate semantic similarity asynchronously."""
        ids, texts = [], []
        for node in topic_nodes_json:
            node_id = node['id']

            # skip if embedding already calculated
            if node_id in self.topic_embed_ref.keys():
                continue
            
            # append topic and description for batch embedding
            topic = node['properties'].get('name', '')
            description = node['properties'].get('description', '')
            if description is not None:
                texts.append(f"{topic} {description}")
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
        
        self.topic_embed_ref.update({key: value for key, value in zip(ids, embeds)})  
    

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

            # skip if embedding already calculated
            if node_id in self.topic_embed_ref.keys():
                continue
            
            # prepare title and abstract for emebdding
            title = node['properties'].get('title')
            abstract = node['properties'].get('abstract')
            if title is not None and abstract is not None:
                texts.append(f"TITLE: {title} \nABSTRACT: {abstract}")
                ids.append(node_id)

        if not texts:
            logging.warning("No paper titles and abstracts found for semantic similarity calculation.")
            return {}

        # calculate embeddings
        logging.info(f"Generating embeddings for {len(texts)} papers...")
        try:
             embeds = await gemini_embedding_async(embed_api_key, embed_model_name, texts, 10) # Pass batch size
        except Exception as e:
             logging.error(f"Embedding generation failed: {e}")
             return {}

        if embeds is None or (len(embeds) != len(texts)):
             logging.error(f"Embedding generation returned unexpected result. Expected {len(texts)} embeddings, got {len(embeds) if embeds else 0}.")
             return {}
        
        # update abstract embedding 
        self.abs_embed_ref.update({key: value for key, value in zip(ids, embeds)})   
    

    async def cal_p2p_similarity(
            self,
            paper_nodes_json: List,
            paper_ids_1: List, 
            paper_ids_2: List,
            similarity_threshold: Optional[float] = 0.7,
            ):
        """calculate paper to paper similarity and filter based on similarity_threshold"""
        # --- 1. Status check ---
        # check if inputpaper id within paper nodes json
        paper_ids = [node['id'] for node in paper_nodes_json]  # all paper ids
        status_1 = bool(set(paper_ids_1).intersection(set(paper_ids)))
        status_2 = bool(set(paper_ids_2).intersection(set(paper_ids)))

        if not status_1 or not status_2:  # ids have no overlap with json info
            logging.error(f"The input {'' if status_1 else '1'} {'' if status_2 else '2'} paper dois have no intersection with current paper json.")
            return []

        # --- 2. Supplement abstract embedding if not exist ---
        input_paper_dois = list(set(paper_ids_1).union(paper_ids_2))  # input paper ids
        if status_1 and status_2:
            # filter those not yet have emebedings
            wo_embeds_json = [node for node in paper_nodes_json 
                              if node['id'] in input_paper_dois
                              and node['id'] not in self.abs_embed_ref]
            
            # calculate embeddings for these nodes
            if wo_embeds_json:
                await self.get_abstract_embeds(wo_embeds_json, self.embed_api_key, self.embed_model_name) 

        # --- 3. Calculate similarities given paper abstract embedding ---
        # Filter embeddings for each list
        embeds_ref_1 = {key: val for key, val in self.abs_embed_ref.items() if key in paper_ids_1}
        embeds_ref_2 = {key: val for key, val in self.abs_embed_ref.items() if key in paper_ids_2}

        ids_1 = list(embeds_ref_1.keys())
        ids_2 = list(embeds_ref_2.keys())
        embed_values_1 = list(embeds_ref_1.values())
        embed_values_2 = list(embeds_ref_2.values())

        # robustness check
        if not embed_values_1 or not embed_values_2:
            logging.warning(f"Cannot calculate similarity. "
                            f"Found {len(embed_values_1)} valid embeddings for list 1 (IDs: {ids_1}) and "
                            f"{len(embed_values_2)} for list 2 (IDs: {ids_2}). "
                            f"Check if input IDs exist and have title/abstract.")
            return [] # Return empty list as no similarity can be calculated

        embeds_1 = np.array(embed_values_1)
        embeds_2 = np.array(embed_values_2)

        # Add logging for shapes *before* the calculation
        logging.info(f"Shape of embeds_1: {embeds_1.shape}")
        logging.info(f"Shape of embeds_2: {embeds_2.shape}")

        logging.info("Calculating similarity matrix...")
        try:
            sim_matrix = semantic_similarity_matrix(embeds_1, embeds_2)
            sim_matrix = np.array(sim_matrix) # Ensure it's a numpy array if not already
        except Exception as e:
            logging.error(f"Similarity matrix calculation failed with embeds_1 shape {embeds_1.shape} and embeds_2 shape {embeds_2.shape}: {e}")
            return [] 

        # --- 4. Process similarity matrix ---
        semantic_similar_pool = process_p2p_sim_data(
            paper_nodes_json = paper_nodes_json,
            paper_ids_1 = ids_1,
            paper_ids_2 = ids_2,
            sim_matrix = sim_matrix, 
            similarity_threshold = similarity_threshold
        )
        return semantic_similar_pool
    

    async def cal_p2t_similarity(
            self,
            paper_nodes_json: List,
            topic_nodes_json: List,
            paper_ids: List,
            topic_ids: List, 
            similarity_threshold: Optional[float] = 0.7,
            ):
        """calculate paper to paper similarity and filter based on similarity_threshold"""
        # --- 1. Status check ---
        # check if inputpaper id within paper nodes json
        all_paper_ids = [node['id'] for node in paper_nodes_json]  # all paper ids
        all_topic_ids = [node['id'] for node in topic_nodes_json]  # all topic ids
        status_1 = bool(set(paper_ids).intersection(set(all_paper_ids)))
        status_2 = bool(set(topic_ids).intersection(set(all_topic_ids)))

        if not status_1 or not status_2:  # ids have no overlap with json info
            logging.error(f"The input {'paper ids' if not status_1 else ''} {'topic ids' if not status_2 else ''} have no intersection with current topic / paper json.")
            return []

        # --- 2. Supplement abstract embedding if not exist ---
        if status_1:
            # paper not yet have emebedings
            paper_wo_embeds_json = [node for node in paper_nodes_json 
                                    if node['id'] in paper_ids
                                    and node['id'] not in self.abs_embed_ref]

            # calculate embeddings for these nodes
            if paper_wo_embeds_json:
                await self.get_abstract_embeds(paper_wo_embeds_json, self.embed_api_key, self.embed_model_name) 
            
        if status_2:
            # topic not yet have emebedings
            topic_wo_embeds_json = [node for node in topic_nodes_json 
                                    if node['id'] in topic_ids
                                    and node['id'] not in self.topic_embed_ref]
            
            # calculate embeddings for these nodes
            if topic_wo_embeds_json:
                await self.get_topic_embeds(topic_wo_embeds_json, self.embed_api_key, self.embed_model_name) 

        # --- 3. Calculate similarities given paper abstract embedding ---
        # Filter embeddings for each list
        embeds_ref_1 = {key: val for key, val in self.abs_embed_ref.items() if key in paper_ids}
        embeds_ref_2 = {key: val for key, val in self.topic_embed_ref.items() if key in topic_ids}

        ids_1 = list(embeds_ref_1.keys())
        ids_2 = list(embeds_ref_2.keys())
        embed_values_1 = list(embeds_ref_1.values())
        embed_values_2 = list(embeds_ref_2.values())

        # robustness check
        if not embed_values_1 or not embed_values_2:
            logging.warning(f"Cannot calculate similarity. "
                            f"Found {len(embed_values_1)} valid embeddings for papers (IDs: {ids_1}) and "
                            f"{len(embed_values_2)} for topics (IDs: {ids_2}). "
                            f"Check if input IDs exist and have title/abstract.")
            return [] # Return empty list as no similarity can be calculated

        embeds_1 = np.array(embed_values_1)
        embeds_2 = np.array(embed_values_2)

        # Add logging for shapes *before* the calculation
        logging.info(f"Shape of embeds_1: {embeds_1.shape}")
        logging.info(f"Shape of embeds_2: {embeds_2.shape}")

        logging.info("Calculating similarity matrix...")
        try:
            sim_matrix = semantic_similarity_matrix(embeds_1, embeds_2)
            sim_matrix = np.array(sim_matrix) # Ensure it's a numpy array if not already
        except Exception as e:
            logging.error(f"Similarity matrix calculation failed with embeds_1 shape {embeds_1.shape} and embeds_2 shape {embeds_2.shape}: {e}")
            return [] 

        # --- 4. Process similarity matrix ---
        semantic_similar_pool = process_p2t_sim_data(
            paper_ids = ids_1,
            topic_ids = ids_2,
            sim_matrix = sim_matrix, 
            similarity_threshold = similarity_threshold
        )
        return semantic_similar_pool
    