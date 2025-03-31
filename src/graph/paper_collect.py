import json
import time
import numpy as np
from collections import Counter
from typing import List, Dict, Optional, Union
from json_repair import repair_json  # https://github.com/mangiucugna/json_repair/  # Consider pros and cons for using this 3rd party lib, and error handling when JSON repair fails

# Assuming these modules are defined elsewhere as per original code structure
from utils.data_process import generate_hash_key
from apis.s2_api import SemanticScholarKit
from models.llms import llm_gen_w_retry
from prompts.query_prompt import keywords_topics_example, keywords_topics_prompt
from models.embedding_models import gemini_embedding_async, semantic_similarity_matrix
from graph.s2_metadata_process import process_paper_metadata, process_citation_metadata, process_related_metadata, process_author_metadata

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PaperSearch:
    """
    A class for exploring academic papers using Semantic Scholar API and LLMs.
    """
    def __init__(
            self,
            research_topic: Optional[str] = None,
            seed_paper_titles: Optional[Union[List[str], str]] = None,
            seed_paper_dois: Optional[Union[List[str], str]] = None,
            llm_api_key: Optional[str] = None, 
            llm_model_name: Optional[str] = None,
            embed_api_key: Optional[str] = None,
            embed_model_name: Optional[str] = None,
            from_dt: Optional[str] = None,   
            to_dt: Optional[str] = None,   
            fields: Optional[List[str]] = None,  
            min_citation_cnt: Optional[int] = 0,   
            institutions: Optional[List[str]] = None,
            journals: Optional[List[str]] = None,    
            author_ids: Optional[List[str]] = None, 
    ):
        """
        Initialize PaperExploration parameters.
        User can input research topic, seed paper title(s), or seed paper doi(s) as a starting point.

        Args: # Added Args section as per feedback
            research_topic (Optional[str]): Research topic to start exploration.
            seed_paper_titles (Optional[Union[List[str], str]]): Seed paper titles to start exploration. Can be a single title (str) or a list of titles (List[str]).
            seed_paper_dois (Optional[Union[List[str], str]]): Seed paper DOIs to start exploration. Can be a single DOI (str) or a list of DOIs (List[str]).
            llm_api_key (Optional[str]): API key for LLM model. Loaded from config.toml if not provided.
            llm_model_name (Optional[str]): Name of LLM model. Loaded from config.toml if not provided.
            embed_api_key (Optional[str]): API key for embedding model. Loaded from config.toml if not provided.
            embed_model_name (Optional[str]): Name of embedding model. Loaded from config.toml if not provided.
            from_dt (Optional[str]): Filter papers published no earlier than this date (YYYY-MM-DD).
            to_dt (Optional[str]): Filter papers published no later than this date (YYYY-MM-DD).
            fields (Optional[List[str]]): List of fields of study to filter papers.
            min_citation_cnt (Optional[int]): Minimum citation count for papers.
            institutions (Optional[List[str]]): List of institutions to restrict paper search (not implemented).
            journals (Optional[List[str]]): List of journals to restrict paper search (not implemented).
            author_ids (Optional[List[str]]): List of author IDs to restrict paper search.
        """
        self.s2 = SemanticScholarKit()
        self.research_topic = research_topic
        self.seed_paper_titles = [seed_paper_titles] if isinstance(seed_paper_titles, str) and seed_paper_titles else seed_paper_titles if isinstance(seed_paper_titles, list) else [] # Handle None and empty string cases more robustly
        self.seed_paper_dois = [seed_paper_dois] if isinstance(seed_paper_dois, str) and seed_paper_dois else seed_paper_dois if isinstance(seed_paper_dois, list) else [] # Handle None and empty string cases more robustly


        # for search result filtering
        self.from_dt = from_dt
        self.to_dt = to_dt
        self.fields = fields # Consistent naming: self.fields
        self.min_citation_cnt = min_citation_cnt
        self.institutions = institutions
        self.journals = journals
        self.author_ids = author_ids

        # llm
        self.llm_api_key = llm_api_key
        self.llm_model_name = llm_model_name
        self.embed_api_key = embed_api_key
        self.embed_model_name = embed_model_name

        # store and process all potential papers information
        self.nodes_json = []
        self.edges_json = []


    def initial_paper_query(
            self,
            limit: Optional[int] = 100,
            from_dt: Optional[str] = None,     # filter publish dt no earlier than
            to_dt: Optional[str] = None,       # filter publish dt no late than
            fields: Optional[List[str]] = None,  # list of field of study, consistent naming as 'fields' (plural)
    ):
        """
        Retrieve papers based on user's input text (research topic, seed paper titles/DOIs).

        Args:
            limit (Optional[int]): Maximum number of papers to retrieve per query. Defaults to 100.
            from_dt (Optional[str]): Filter papers published no earlier than this date (YYYY-MM-DD). Overrides class-level `from_dt` if provided.
            to_dt (Optional[str]): Filter papers published no later than this date (YYYY-MM-DD). Overrides class-level `to_dt` if provided.
            fields (Optional[List[str]]): List of fields of study to filter papers. Overrides class-level `fields` if provided.
        """
        # retrieve paper metadata from s2
        seed_paper_metadata, searched_paper_metadata = [], [] 
        current_fields = fields if fields is not None else self.fields
        current_from_dt = from_dt if from_dt is not None else self.from_dt 
        current_to_dt = to_dt if to_dt is not None else self.to_dt

        if self.seed_paper_dois:
            s2_paper_metadata = self.s2.search_paper_by_ids(id_list=self.seed_paper_dois, fields=current_fields)
            seed_paper_metadata.extend(s2_paper_metadata)
            time.sleep(5)

        if self.seed_paper_titles and len(self.seed_paper_titles) > 0:
            for title in self.seed_paper_titles:
                s2_paper_metadata = self.s2.search_paper_by_keywords(query=title, fields=current_fields, limit=limit)
                if s2_paper_metadata: # Check if s2_paper_metadata is not empty to avoid IndexError
                    seed_paper_metadata.append(s2_paper_metadata[0])
                    searched_paper_metadata.extend(s2_paper_metadata[1:]) 
                    time.sleep(5)

        if self.research_topic:
            s2_paper_metadata = self.s2.search_paper_by_keywords(query=self.research_topic, fields=current_fields, limit=limit)
            searched_paper_metadata.extend(s2_paper_metadata) 
            time.sleep(5)

        # store seed paper metadata
        if len(seed_paper_metadata) > 0:
            # convert to standard format, be aware the output include nodes and edges
            seed_papermetadata_json = process_paper_metadata(
                s2_paper_metadata=seed_paper_metadata,
                from_dt=current_from_dt,
                to_dt=current_to_dt,
                fields=current_fields)

            # iterate processed paper metadata, and store information to nodes_json and edge_json separately
            for item in seed_papermetadata_json:
                if item['type'] == 'node':
                    if item['labels'] == ['Paper']:
                        item['properties']['from_seed'] = True
                        item['properties']['is_complete'] = True
                    self.nodes_json.append(item)
                elif item['type'] == 'relationship':
                    self.edges_json.append(item)

        # store searched paper metadata
        if len(searched_paper_metadata) > 0:
            # convert to standard format, be aware the output include nodes and edges
            searched_papermetadata_json = process_paper_metadata( 
                s2_paper_metadata=searched_paper_metadata, 
                from_dt=current_from_dt,
                to_dt=current_to_dt,
                fields=current_fields)

            # iterate processed paper metadata, and store information to nodes_json and edge_json separately
            for item in searched_papermetadata_json:
                if item['type'] == 'node':
                    if item['labels'] == ['Paper']:
                        item['properties']['from_search'] = True
                        item['properties']['is_complete'] = True
                    self.nodes_json.append(item)
                elif item['type'] == 'relationship':
                    self.edges_json.append(item)


    def get_author_info(
            self, 
            author_ids,
            from_dt: Optional[str] = None,     # filter publish dt no earlier than
            to_dt: Optional[str] = None,       # filter publish dt no late than
            fields: Optional[List[str]] = None,  # list of field of study
            ):
        # author_ids = []
        # for item in seed_paper_metadata:
        #     authors = item.get('authors', [])[0:5]
        #     author_ids.extend([x['authorId'] for x in authors if x['authorId']])
        current_fields = fields if fields is not None else self.fields
        current_from_dt = from_dt if from_dt is not None else self.from_dt 
        current_to_dt = to_dt if to_dt is not None else self.to_dt

        # retrieve author metadata from s2
        authros_info = self.s2.search_author_by_ids(author_ids=author_ids, fields=fields, with_abstract=True)
        time.sleep(5)

        # convert to standard format, be aware the output include nodes and edges
        s2_author_meta_json = process_author_metadata(
            s2_author_metadata=authros_info,
            from_dt=current_from_dt,
            to_dt=current_to_dt,
            fields=current_fields
        )

        # iterate processed paper metadata, and store information to nodes_json and edge_json separately
        for item in s2_author_meta_json:
            if item['type'] == 'node':
                if item['labels'] == ['Paper']:
                    item['properties']['from_same_author'] = True
                    item['properties']['is_complete'] = True
                elif item['labels'] == ['Author']:
                    item['properties']['is_complete'] = True
                self.nodes_json.append(item)
            elif item['type'] == 'relationship':
                self.edges_json.append(item)


    def get_cited_papers(
            self,
            paper_doi,
            limit: Optional[int] = 100,
            from_dt: Optional[str] = None,     # filter publish dt no earlier than
            to_dt: Optional[str] = None,       # filter publish dt no late than
            fields: Optional[List[str]] = None,  # list of field of study
    ):
        """
        Get papers cited by the paper with the given DOI.

        Args:
            paper_doi (str): DOI of the paper to find cited papers for.
            limit (Optional[int]): Maximum number of cited papers to retrieve. Defaults to 100.
            from_dt (Optional[str]): Filter papers published no earlier than this date (YYYY-MM-DD). Overrides class-level `from_dt` if provided.
            to_dt (Optional[str]): Filter papers published no later than this date (YYYY-MM-DD). Overrides class-level `to_dt` if provided.
            fields (Optional[List[str]]): List of fields of study to filter papers. Overrides class-level `fields` if provided.
        """
        # retrieve cited paper metadata from s2
        current_fields = fields if fields is not None else self.fields 
        current_from_dt = from_dt if from_dt is not None else self.from_dt 
        current_to_dt = to_dt if to_dt is not None else self.to_dt

        s2_citedpaper_metadata = self.s2.get_s2_cited_papers(paper_doi, fields=current_fields, limit=limit, with_abstract=True)
        time.sleep(5)

        # convert to standard format, be aware the output include nodes and edges
        s2_citedpapermeta_json = process_citation_metadata(
            original_paper_doi=paper_doi,
            s2_citation_metadata=s2_citedpaper_metadata,
            citation_type='citedPaper',
            from_dt=current_from_dt,
            to_dt=current_to_dt,
            fields=current_fields)

        # iterate processed paper metadata, and store information to nodes_json and edge_json separately
        for item in s2_citedpapermeta_json:
            if item['type'] == 'node':
                if item['labels'] == ['Paper']:
                    item['properties']['from_cited'] = True
                    item['properties']['is_complete'] = True
                self.nodes_json.append(item)
            elif item['type'] == 'relationship':
                self.edges_json.append(item)


    def get_citing_papers(
            self,
            paper_doi,
            limit: Optional[int] = 100,
            from_dt: Optional[str] = None,     # filter publish dt no earlier than
            to_dt: Optional[str] = None,       # filter publish dt no late than
            fields: Optional[List[str]] = None,  # list of field of study
    ):
        """
        Retrieve papers that cite the paper with the given DOI.

        Args:
            paper_doi (str): DOI of the paper to find citing papers for.
            limit (Optional[int]): Maximum number of citing papers to retrieve. Defaults to 100.
            from_dt (Optional[str]): Filter papers published no earlier than this date (YYYY-MM-DD). Overrides class-level `from_dt` if provided.
            to_dt (Optional[str]): Filter papers published no later than this date (YYYY-MM-DD). Overrides class-level `to_dt` if provided.
            fields (Optional[List[str]]): List of fields of study to filter papers. Overrides class-level `fields` if provided.
        """
        # retrieve citing paper metadata from s2
        current_fields = fields if fields is not None else self.fields 
        current_from_dt = from_dt if from_dt is not None else self.from_dt 
        current_to_dt = to_dt if to_dt is not None else self.to_dt 

        s2_citingpaper_metadata = self.s2.get_s2_citing_papers(paper_doi, fields=current_fields, limit=limit, with_abstract=True)
        time.sleep(5)

        # convert to standard format, be aware the output include nodes and edges
        s2_citingpapermetadata_json = process_citation_metadata(
            original_paper_doi=paper_doi,
            s2_citation_metadata=s2_citingpaper_metadata,
            citation_type='citingPaper',
            from_dt=current_from_dt,
            to_dt=current_to_dt,
            fields=current_fields)

        # iterate processed paper metadata, and store information to nodes_json and edge_json separately
        for item in s2_citingpapermetadata_json:
            if item['type'] == 'node':
                if item['labels'] == ['Paper']:
                    item['properties']['from_citing'] = True
                    item['properties']['is_complete'] = True
                self.nodes_json.append(item)
            elif item['type'] == 'relationship':
                self.edges_json.append(item)

    def get_recommend_papers(
            self,
            paper_dois: Union[List[str], str], # Revised type hint to use Union
            limit: Optional[int] = 100,
            from_dt: Optional[str] = None,     # filter publish dt no earlier than
            to_dt: Optional[str] = None,       # filter publish dt no late than
            fields: Optional[List[str]] = None,  # list of field of study
    ):
        """
        Retrieve papers recommended by Semantic Scholar based on a list of paper DOIs.

        Args:
            paper_dois (Union[List[str], str]): DOI(s) of papers to get recommendations for. Can be a single DOI (str) or a list of DOIs (List[str]).
            limit (Optional[int]): Maximum number of recommended papers to retrieve. Defaults to 100.
            from_dt (Optional[str]): Filter papers published no earlier than this date (YYYY-MM-DD). Overrides class-level `from_dt` if provided.
            to_dt (Optional[str]): Filter papers published no later than this date (YYYY-MM-DD). Overrides class-level `to_dt` if provided.
            fields (Optional[List[str]]): List of fields of study to filter papers. Overrides class-level `fields` if provided.
        """
        # retrieve recommended paper metadata from s2
        current_fields = fields if fields is not None else self.fields # Use provided fields if available, otherwise use class-level fields
        current_from_dt = from_dt if from_dt is not None else self.from_dt # Use provided from_dt if available, otherwise use class-level from_dt
        current_to_dt = to_dt if to_dt is not None else self.to_dt # Use provided to_dt if available, otherwise use class-level to_dt

        if isinstance(paper_dois, str):
            paper_dois = [paper_dois]
        s2_recommended_metadata = self.s2.get_s2_recommended_papers(positive_paper_ids=paper_dois, fields=current_fields, limit=limit, with_abstract=True)
        time.sleep(5)

        # convert to standard format, be aware the output include nodes and edges
        s2_recpapermetadata_json = process_paper_metadata(
            s2_paper_metadata=s2_recommended_metadata,
            from_dt=current_from_dt,
            to_dt=current_to_dt,
            fields=current_fields)

        # iterate processed paper metadata, and store information to nodes_json and edge_json separately
        for item in s2_recpapermetadata_json:
            if item['type'] == 'node':
                if item['labels'] == ['Paper']:
                    item['properties']['from_recommended'] = True
                    item['properties']['is_complete'] = True
                self.nodes_json.append(item)
            elif item['type'] == 'relationship':
                self.edges_json.append(item)


    def llm_gen_related_topics(self, seed_paper_ids):
        """generate realted topic for further search"""
        # get domain and paper title, abstracts
        domains, seed_paper_texts = [], []
        node_ids = [x['id'] for x in self.nodes_json]
        for paper_id in seed_paper_ids:
            idx = node_ids.index(paper_id)
            item = self.nodes_json[idx]
            title = item['properties'].get('title')
            abstract = item['properties'].get('abstract')
            domain = item['properties'].get('fieldsOfStudy')
            info = f"<paper> {title}\n{abstract} </paper>"
            seed_paper_texts.append(info)
            domains.extend(domain)
        domain = Counter(domains).most_common(1)[0][0]

        # llm propose search queries
        qa_prompt = keywords_topics_prompt.format(
                domain = domain,
                example_json = keywords_topics_example,
                input_text = "\n\n".join(seed_paper_texts)
            )
        keywords_topics_info = llm_gen_w_retry(self.llm_api_key, self.llm_model_name, qa_prompt, sys_prompt=None, temperature=0.6)
            
        # extract keywords, topics, queries
        try:
            keywords_topics_json = json.loads(repair_json(keywords_topics_info)) # Use try-except to handle potential JSON repair/parsing errors
        except json.JSONDecodeError as e:
            logging.error(f"JSON Repair or Decode Error: {e}. Original LLM output: {keywords_topics_info}")
            keywords_topics_json = {} # Initialize to empty dict to avoid further errors, handle gracefully later if needed
        print(keywords_topics_json)

        # llm propose search queries
        query_topics = keywords_topics_json.get('queries')
        for topic in query_topics: 
            topic_hash_id = generate_hash_key(topic)
            if topic_hash_id not in node_ids:
                topic_node = {
                    'type': 'node',
                    'id': topic_hash_id,
                    'labels': ['Topic'],
                    'properties': {'topic_hash_id': topic_hash_id, 'topic_name': topic, 'hash_method': 'hashlib.sha256'}}
                self.nodes_json.append(topic_node)

            for paper_id in seed_paper_ids:
                edge_ids = [(x['startNodeId'],x['endNodeId'])  for x in self.edges_json if x['type']=='relationship']
                if (paper_id, topic_hash_id) not in edge_ids:
                    paper_topic_relationship = {
                        "type": "relationship",
                        "relationshipType": "DISCUSS",
                        "startNodeId": paper_id,
                        "endNodeId": topic_hash_id,
                        "properties": {}}
                    self.edges_json.append(paper_topic_relationship)

        return keywords_topics_json


    def get_related_papers(
            self,
            topic_json: Dict,
            limit: Optional[int] = 100,
            from_dt: Optional[str] = None,     # filter publish dt no earlier than
            to_dt: Optional[str] = None,       # filter publish dt no late than
            fields: Optional[List[str]] = None,  # list of field of study
    ):
        """
        Further expand paper scope by leveraging deep search with LLMs to generate related search queries.

        Args:
            domain (str): Research domain or area. Used by LLM to generate relevant queries.
            input_text (str): User input text related to the research topic. Used by LLM to generate relevant queries.
            limit (Optional[int]): Maximum number of papers to retrieve per query. Defaults to 100.
            from_dt (Optional[str]): Filter papers published no earlier than this date (YYYY-MM-DD). Overrides class-level `from_dt` if provided.
            to_dt (Optional[str]): Filter papers published no later than this date (YYYY-MM-DD). Overrides class-level `to_dt` if provided.
            fields (Optional[List[str]]): List of fields of study to filter papers. Overrides class-level `fields` if provided.
        """
        # retrieve related papers using LLM-generated queries
        current_fields = fields if fields is not None else self.fields 
        current_from_dt = from_dt if from_dt is not None else self.from_dt 
        current_to_dt = to_dt if to_dt is not None else self.to_dt 

        # llm propose search queries
        queries = topic_json.get('queries')

        if queries: # Check if queries is not None and not empty list to avoid errors
            for query in queries:
                s2_paper_metadata = self.s2.search_paper_by_keywords(query, fields=current_fields, limit=limit)
                time.sleep(5)

                # convert to standard format, be aware the output include nodes and edges
                s2_papermeta_json = process_related_metadata(
                    s2_related_metadata=s2_paper_metadata,
                    from_dt=current_from_dt,
                    to_dt=current_to_dt,
                    fields=current_fields)

                # iterate processed paper metadata, and store information to nodes_json and edge_json separately
                for item in s2_papermeta_json:
                    if item['type'] == 'node':
                        if item['labels'] == ['Paper']:
                            item['properties']['from_related_topics'] = True
                            item['properties']['is_complete'] = True
                        self.nodes_json.append(item)
                    elif item['type'] == 'relationship':
                        self.edges_json.append(item)


    async def cal_semantic_similarity(self, paper_nodes_json):
        """
        Add semantic similarity relationships (edges) between paper nodes based on title and abstract.
        This method calculates semantic similarity using embeddings generated by 'texts_embed_gen' and
        adds 'SIMILAR_TO' relationships with 'weight' property representing the similarity score.
        """
        # existing edge ids
        semantic_similar_pool = []

        # get paper publish date
        publish_dt_ref = {x['id']:x['properties'].get('publicationDate')
             for x in paper_nodes_json if x['properties'].get('publicationDate') is not None}
        
        # extract title and abstract from paper nodes json
        ids, texts = [], []
        for node in paper_nodes_json:
            id = node['id']
            title = node['properties'].get('title')
            abstract = node['properties'].get('abstract')
            if title is not None and abstract is not None:
                texts.append(f"TITLE: {title} \nABSTRACT: {abstract}")
                ids.append(id)
        paper_info_ref = {key: value for key, value in zip(ids, texts)}

        if not texts: # Handle case where no texts are available for embedding
            logging.warning("No paper titles and abstracts found for semantic similarity calculation.")
            return

        # then calculate semantic similarities between the texts
        embeds = await gemini_embedding_async(self.embed_api_key, self.embed_model_name, texts, 10) # Assuming texts_embed_gen is an async function for IO-bound operations

        # calculate similarity matrix
        sim_matrix = semantic_similarity_matrix(embeds, embeds)
        sim_matrix = np.array(sim_matrix)

        # iterate similarity matrix to add similarity relationships
        rows, cols = sim_matrix.shape
        for i in range(rows):
            publish_dt_i = publish_dt_ref.get(ids[i])
            for j in range(cols):
                publish_dt_j = publish_dt_ref.get(ids[j])
                sim = sim_matrix[i, j]
                if i != j:
                    if publish_dt_i < publish_dt_j:
                        start_node_id = ids[i]
                        end_node_id = ids[j]
                    else:
                        start_node_id = ids[j]
                        end_node_id = ids[i]
                    edge = {
                        "type": "relationship",
                        "relationshipType": "SIMILAR_TO",
                        "startNodeId": start_node_id,
                        "endNodeId": end_node_id,
                        "properties": {'source': 'semantic similarity', 'weight': round(sim, 4),
                                       'texts':(paper_info_ref.get(start_node_id), paper_info_ref.get(end_node_id))}
                        }
                    semantic_similar_pool.append(edge)
        return semantic_similar_pool
    
    def init_collect(
            self,
            limit=50, 
            from_dt='2022-01-01', 
            to_dt='2025-03-13',
            fields=None,
            ):
        self.initial_paper_query(limit=limit, from_dt=from_dt, to_dt=to_dt, fields=fields)

    async def collect(
            self, 
            seed_paper_dois: List[str],
            with_author: Optional[bool] = True,
            with_recommend: Optional[bool] = True,
            with_expanded_search: Optional[bool] = True, 
            add_semantic_similarity: Optional[bool] = True,
            similarity_threshold: Optional[float] = 0.7,
            limit=50, 
            from_dt='2022-01-01', 
            to_dt='2025-03-13',
            fields=None,):
        # get seed papers' author information
        if with_author:
            # first get author for seed paper doi
            author_ids = []
            for item in self.nodes_json:
                if item['labels'] == ["Paper"] and item['id'] in seed_paper_dois:
                    authors = item['properties'].get('authors')
                    if isinstance(authors, list):
                        paper_author_ids = [x['authorId'] for x in authors if x.get('authorId') is not None]
                        author_ids.extend(paper_author_ids)
            # get author ids with complete information
            complete_author_ids = [x for x in self.nodes_json if x['labels'] == ["Author"] and x['properties'].get('is_complete') == True]
            # filter author ids with complete information
            author_ids_filtered = list(set(author_ids) - set(complete_author_ids))
            # get author information
            self.get_author_info(author_ids_filtered, from_dt, to_dt, fields)

        # get citation for seed papers
        for paper_doi in seed_paper_dois:
            self.get_cited_papers(paper_doi, limit, from_dt, to_dt, fields) 
            time.sleep(5)
            self.get_citing_papers(paper_doi, limit, from_dt, to_dt, fields) 
            time.sleep(5)

        # get recommended papers
        if with_recommend:
            self.get_recommend_papers(seed_paper_dois, limit, from_dt, to_dt, fields)

        # get expanded search by llm detected topics
        if with_expanded_search:
            # first use LLM to generate research topics based on seed papers
            keywords_topics_json = self.llm_gen_related_topics(self, seed_paper_dois)
            # then conduct search for related papers on each research topics 
            self.get_related_papers(keywords_topics_json, limit, from_dt, to_dt, fields)

        # add semantic similarity relationship
        if add_semantic_similarity:
            paper_nodes_json = [x for x in self.nodes_json if x['labels'] == ["Paper"]] 
            # calculate semantic similarity
            semantic_similar_pool = await self.cal_semantic_similarity(paper_nodes_json)
            # filter similarity score by thrshold
            semantic_similar_relationship = [x for x in semantic_similar_pool if x['properties'].get('weight') > similarity_threshold]
            self.edges_json.append(semantic_similar_relationship)
        

