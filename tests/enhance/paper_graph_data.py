import re
import json
import time
import toml
from typing import List, Dict, Optional, Union
from json_repair import repair_json  # https://github.com/mangiucugna/json_repair/  # Consider pros and cons for using this 3rd party lib, and error handling when JSON repair fails

# Assuming these modules are defined elsewhere as per original code structure
from apis.s2_api import SemanticScholarKit
from utils.data_process import rename_key_in_dict, remove_kth_element, remove_key_values, filter_and_reorder_dict


class PaperExploration:
    """
    A class for exploring academic papers using Semantic Scholar API and LLMs.
    """
    def __init__(
            self,
            research_topic: Optional[str] = None,
            seed_paper_titles: Optional[Union[List[str], str]] = None, # Revised type hint to use Union
            seed_paper_dois: Optional[Union[List[str], str]] = None,   # Revised type hint to use Union
            llm_api_key: Optional[str] = config_param.get('models', {}).get('llm', {}).get('api_key'), # Use .get to avoid KeyError
            llm_model_name: Optional[str] = config_param.get('models', {}).get('llm', {}).get('model_name'), # Use .get to avoid KeyError
            embed_api_key: Optional[str] = config_param.get('models', {}).get('embed', {}).get('api_key'), # Use .get to avoid KeyError
            embed_model_name: Optional[str] = config_param.get('models', {}).get('embed', {}).get('model_name'), # Use .get to avoid KeyError
            from_dt: Optional[str] = None,     # filter publish dt no earlier than
            to_dt: Optional[str] = None,       # filter publish dt no late than
            fields: Optional[List[str]] = None,  # list of field of study, consistent naming as 'fields' (plural)
            min_citation_cnt: Optional[int] = 0,   # citation count no less than
            institutions: Optional[List[str]] = None, # restricted to list of institutions, to be implemented
            journals: Optional[List[str]] = None,     # restricted to list of journals, to be implemented
            author_ids: Optional[List[str]] = None,   # restricted to list of authors' ids
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
            searched_paper_metadata.extend(s2_paper_metadata) # Renamed 'srched_paper_metadata' to 'searched_paper_metadata'
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
            # nodes need to dedup, however, edges do not need to dedup
            node_id_pool = [x['id'] for x in self.nodes_json]
            for item in seed_papermetadata_json:
                if item['type'] == 'node':
                    curr_node_id = item['id']
                    if curr_node_id not in node_id_pool:  # for new node
                        item['properties']['source'] = ['Seed']
                        item['properties']['sourceDesc'] = ['Original seed papers']
                        self.nodes_json.append(item)
                    else:   # for existing node
                        idx = node_id_pool.index(curr_node_id)
                        if isinstance(self.nodes_json[idx].get('source'), list):
                            self.nodes_json[idx]['source'].append('Seed')
                        else:
                            self.nodes_json[idx]['source'] = ['Seed']
                        if isinstance(self.nodes_json[idx].get('sourceDesc'), list):
                            self.nodes_json[idx]['sourceDesc'].append('Original seed papers')
                        else:
                            self.nodes_json[idx]['sourceDesc'] = ['Original seed papers']

                elif item['type'] == 'relationship':
                    # we aim to construct multigraph, which allows multiple edges between any pair of nodes
                    self.edges_json.append(item)

        # store searched paper metadata
        if len(searched_paper_metadata) > 0: # Renamed 'srched_paper_metadata' to 'searched_paper_metadata'
            # convert to standard format, be aware the output include nodes and edges
            searched_papermetadata_json = process_paper_metadata( # Renamed 'srched_papermetadata_json' to 'searched_papermetadata_json'
                s2_paper_metadata=searched_paper_metadata, # Renamed 'srched_paper_metadata' to 'searched_paper_metadata'
                from_dt=current_from_dt,
                to_dt=current_to_dt,
                fields=current_fields)

            # iterate processed paper metadata, and store information to nodes_json and edge_json separately
            # nodes need to dedup, however, edges do not need to dedup
            node_id_pool = [x['id'] for x in self.nodes_json]
            for idx, item in enumerate(searched_papermetadata_json): # Renamed 'srched_papermetadata_json' to 'searched_papermetadata_json'
                if item['type'] == 'node':
                    curr_node_id = item['id']
                    if curr_node_id not in node_id_pool:  # for new node
                        item['properties']['source'] = ['InitialSearch']
                        item['properties']['sourceDesc'] = ['Search from S2 based on user input']
                        self.nodes_json.append(item)
                    else:   # for existing node
                        idx = node_id_pool.index(curr_node_id)
                        if isinstance(self.nodes_json[idx].get('source'), list):
                            self.nodes_json[idx]['source'].append('InitialSearch')
                        else:
                            self.nodes_json[idx]['source'] = ['InitialSearch']
                        if isinstance(self.nodes_json[idx].get('sourceDesc'), list):
                            self.nodes_json[idx]['sourceDesc'].append('Search from S2 based on user input')
                        else:
                            self.nodes_json[idx]['sourceDesc'] = ['Search from S2 based on user input']

                elif item['type'] == 'relationship':
                    # we aim to construct multigraph, which allows multiple edges between any pair of nodes
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

        s2_citedpaper_metadata = self.s2.get_s2_cited_papers(paper_doi, fields=current_fields, limit=limit)
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
        # nodes need to dedup, however, edges do not need to dedup
        node_id_pool = [x['id'] for x in self.nodes_json]
        for item in s2_citedpapermeta_json:
            if item['type'] == 'node':
                curr_node_id = item['id']
                if curr_node_id not in node_id_pool:  # for new node
                    item['properties']['source'] = ['CitedPaper']
                    item['properties']['sourceDesc'] = [f'cited by {paper_doi}']
                    self.nodes_json.append(item)
                else:   # for existing node
                    idx = node_id_pool.index(curr_node_id)
                    if isinstance(self.nodes_json[idx].get('source'), list):
                        self.nodes_json[idx]['source'].append('CitedPaper')
                    else:
                        self.nodes_json[idx]['source'] = ['CitedPaper']
                    if isinstance(self.nodes_json[idx].get('sourceDesc'), list):
                        self.nodes_json[idx]['sourceDesc'].append(f'cited by {paper_doi}')
                    else:
                        self.nodes_json[idx]['sourceDesc'] = [f'cited by {paper_doi}']

            elif item['type'] == 'relationship':
                # we aim to construct multigraph, which allows multiple edges between any pair of nodes
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

        s2_citingpaper_metadata = self.s2.get_s2_citing_papers(paper_doi, fields=current_fields, limit=limit)
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
        # nodes need to dedup, however, edges do not need to dedup
        node_id_pool = [x['id'] for x in self.nodes_json]
        for item in s2_citingpapermetadata_json:
            if item['type'] == 'node':
                curr_node_id = item['id']
                if curr_node_id not in node_id_pool:  # for new node
                    item['properties']['source'] = ['CitingPaper']
                    item['properties']['sourceDesc'] = [f"citing {paper_doi}"]
                    self.nodes_json.append(item)
                else:   # for existing node
                    idx = node_id_pool.index(curr_node_id)
                    if isinstance(self.nodes_json[idx].get('source'), list):
                        self.nodes_json[idx]['source'].append('CitingPaper')
                    else:
                        self.nodes_json[idx]['source'] = ['CitingPaper']
                    if isinstance(self.nodes_json[idx].get('sourceDesc'), list):
                        self.nodes_json[idx]['sourceDesc'].append(f"citing {paper_doi}")
                    else:
                        self.nodes_json[idx]['sourceDesc'] = [f"citing {paper_doi}"]

            elif item['type'] == 'relationship':
                # we aim to construct multigraph, which allows multiple edges between any pair of nodes
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
        s2_recommended_metadata = self.s2.get_s2_recommended_papers(positive_paper_ids=paper_dois, fields=current_fields, limit=limit)
        time.sleep(5)

        # convert to standard format, be aware the output include nodes and edges
        s2_recpapermetadata_json = process_paper_metadata(
            s2_paper_metadata=s2_recommended_metadata,
            from_dt=current_from_dt,
            to_dt=current_to_dt,
            fields=current_fields)

        # iterate processed paper metadata, and store information to nodes_json and edge_json separately
        # nodes need to dedup, however, edges do not need to dedup
        node_id_pool = [x['id'] for x in self.nodes_json]
        for item in s2_recpapermetadata_json:

            if item['type'] == 'node':
                curr_node_id = item['id']
                if curr_node_id not in node_id_pool:  # for new node
                    item['properties']['source'] = ['RecommendedPaper']
                    item['properties']['sourceDesc'] = [f"recommended by s2 given papers {','.join(paper_dois)}"] # Corrected typo "recomend" to "recommended"
                    self.nodes_json.append(item)
                else:   # for existing node
                    idx = node_id_pool.index(curr_node_id)
                    if isinstance(self.nodes_json[idx].get('source'), list):
                        self.nodes_json[idx]['source'].append('RecommendedPaper')
                    else:
                        self.nodes_json[idx]['source'] = ['RecommendedPaper']
                    if isinstance(self.nodes_json[idx].get('sourceDesc'), list):
                        self.nodes_json[idx]['sourceDesc'].append(f"recommended by s2 given papers {','.join(paper_dois)}") # Corrected typo "recomend" to "recommended"
                    else:
                        self.nodes_json[idx]['sourceDesc'] = [f"recommended by s2 given papers {','.join(paper_dois)}"] # Corrected typo "recomend" to "recommended"

            elif item['type'] == 'relationship':
                # we aim to construct multigraph, which allows multiple edges between any pair of nodes
                self.edges_json.append(item)


    def get_related_papers(
            self,
            domain,
            input_text,
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
        keywords_topics_info = search_query_gen(domain, input_text, self.llm_api_key, self.llm_model_name)
        # extract keywords, topics, queries
        try:
            keywords_topics_json = json.loads(repair_json(keywords_topics_info)) # Use try-except to handle potential JSON repair/parsing errors
        except json.JSONDecodeError as e:
            logging.error(f"JSON Repair or Decode Error: {e}. Original LLM output: {keywords_topics_info}")
            keywords_topics_json = {} # Initialize to empty dict to avoid further errors, handle gracefully later if needed
        print(keywords_topics_json)

        field_of_study = keywords_topics_json.get('field_of_study')
        keywords_and_topics = keywords_topics_json.get('keywords_and_topics')
        tags = keywords_topics_json.get('tags')
        queries = keywords_topics_json.get('queries')

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
                # nodes need to dedup, however, edges do not need to dedup
                node_id_pool = [x['id'] for x in self.nodes_json]
                for item in s2_papermeta_json:
                    if item['type'] == 'node':
                        curr_node_id = item['id']
                        if curr_node_id not in node_id_pool:  # for new node
                            item['properties']['source'] = ['LLMQuery']
                            item['properties']['sourceDesc'] = [query]
                            self.nodes_json.append(item)
                        else:   # for existing node
                            idx = node_id_pool.index(curr_node_id)
                            if isinstance(self.nodes_json[idx].get('source'), list):
                                self.nodes_json[idx]['source'].append('LLMQuery')
                            else:
                                self.nodes_json[idx]['source'] = ['LLMQuery']
                            if isinstance(self.nodes_json[idx].get('sourceDesc'), list):
                                self.nodes_json[idx]['sourceDesc'].append(query)
                            else:
                                self.nodes_json[idx]['sourceDesc'] = [query]

                    elif item['type'] == 'relationship':
                        # we aim to construct multigraph, which allows multiple edges between any pair of nodes
                        self.edges_json.append(item)


    async def add_semantic_relationship(self, paper_nodes_json):
        """
        Add semantic similarity relationships (edges) between paper nodes based on title and abstract.
        This method calculates semantic similarity using embeddings generated by 'texts_embed_gen' and
        adds 'SIMILAR_TO' relationships with 'weight' property representing the similarity score.
        """
        # existing edge ids
        edges_id_pool = [(x['startNodeId'], x['endNodeId']) for x in self.edges_json]

        # first extract title and abstract from paper nodes json
        ids, texts = [], []
        for node in paper_nodes_json:
            id = node['id']
            title = node['properties'].get('title')
            abstract = node['properties'].get('abstract')
            if title is not None and abstract is not None:
                texts.append(f"TITLE: {title} \nABSTRACT: {abstract}")
                ids.append(id)

        if not texts: # Handle case where no texts are available for embedding
            logging.warning("No paper titles and abstracts found for semantic similarity calculation.")
            return

        # then calculate semantic similarities between the texts
        embeds = await texts_embed_gen(texts, self.embed_api_key, self.embed_model_name) # Assuming texts_embed_gen is an async function for IO-bound operations

        # calculate similarity matrix
        sim_matrix = semantic_similarity_gen(embeds, embeds)

        # iterate similarity matrix to add similarity relationships
        rows, cols = sim_matrix.shape
        for i in range(rows):
            for j in range(cols):
                sim = sim_matrix[i, j]
                if i != j:
                    # if edge exist, then update weight
                    if (ids[i], ids[j]) in edges_id_pool:
                        pos = edges_id_pool.index((ids[i], ids[j]))
                        self.edges_json[pos]['properties']['weight'] = round(sim, 4)
                    # if not exit, then generate new edge
                    else:
                        edge = {
                            "type": "relationship",
                            "relationshipType": "SIMILAR_TO",
                            "startNodeId": ids[i],
                            "endNodeId": ids[j],
                            "properties": {'source': 'semantic similarity', 'weight': round(sim, 4)}
                            }
                        self.edges_json.append(edge)