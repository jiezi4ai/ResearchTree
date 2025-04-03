from typing import List, Dict, Optional
import json
from json_repair import repair_json  # https://github.com/mangiucugna/json_repair/

from apis.s2_api import SemanticScholarKit
from paper_metadata_process import process_paper_metadata, process_citation_metadata, process_related_metadata
from topic_tree_gen import search_query_gen, texts_embed_gen, semantic_similarity_gen


class PaperExploration:
    def __init__(
            self, 
            input_text, 
            llm_api_key,
            llm_model_name,
            embed_api_key,
            embed_model_name,
            from_dt: Optional[str] = None,   # filter publish dt no earlier than
            to_dt: Optional[str] = None,   # filter publish dt no late than
            field: Optional[List[str]] = None,  # list of field of study
            min_citation_cnt: Optional[int] = 0,  # citation count no less than
            institutions: Optional[List[str]] = None,  # restrcted to list of institutions, to be implemented
            journals: Optional[List[str]] = None,  # restrcted to list of journals, to be implemented
            author_ids: Optional[List[str]] = None,  # restrcted to list of authors' ids
    ):
        self.s2 = SemanticScholarKit()
        self.user_query = input_text  # use as initial query

        # for search result filtering
        self.from_dt = from_dt
        self.to_dt = to_dt
        self.field = field
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
        self.node_id_pool = set()
        self.nodes_json = []
        self.edges_json = []


    def initial_paper_query(
            self, 
            limit:Optional[int]=100,
            from_dt: Optional[str] = None,   # filter publish dt no earlier than
            to_dt: Optional[str] = None,   # filter publish dt no late than
            field: Optional[List[str]] = None,  # list of field of study
            min_citation_cnt: Optional[int] = 0,  # citation count no less than
            institutions: Optional[List[str]] = None,  # restrcted to list of institutions, to be implemented
            journals: Optional[List[str]] = None,  # restrcted to list of journals, to be implemented
            author_ids: Optional[List[str]] = None,  # restrcted to list of authors' ids      
            ):
        """retrieve papers based on user's input text"""
        # retrieve paper metadata from s2
        s2_paper_metadata = self.s2.search_paper_by_keywords(query=self.user_query, fields_of_study=field, limit=limit)
        
        # convert to standard format, be aware the output include nodes and edges 
        s2_papermeta_json = process_paper_metadata(
            s2_paper_metadata,
            from_dt,
            to_dt,
            field,
            min_citation_cnt,
            institutions,
            journals,
            author_ids)

        # iterate processed paper metadata, and store information to nodes_json and edge_json separately
        # nodes need to dedup, however, edges do not need to dedup
        node_id_pool = [x['id'] for x in self.nodes_json]
        for item in s2_papermeta_json:
            if item['type'] == 'node':
                curr_node_id = item['id']
                if curr_node_id not in node_id_pool:  # for new node
                    item['properties']['source'] = ['UserQuery']
                    item['properties']['sourceDesc'] = [self.user_query]
                    self.nodes_json.append(item)
                else:   # for existing node
                    idx = node_id_pool.index(curr_node_id)
                    if type(self.nodes_json[idx].get('source')) == list:
                        self.nodes_json[idx]['source'].append('UserQuery')
                    else:
                        self.nodes_json[idx]['source'] = ['UserQuery']
                    if type(self.nodes_json[idx].get('sourceDesc')) == list:
                        self.nodes_json[idx]['sourceDesc'].append(self.user_query)
                    else:
                        self.nodes_json[idx]['sourceDesc'] = [self.user_query]
            
            elif item['type'] == 'relationship':
                # we aim to construct multigraph, which allows multiple edges between any pair of nodes
                self.edges_json.append(item)


    def get_cited_papers(
            self, 
            paper_id, 
            limit:Optional[int]=100,
            from_dt: Optional[str] = None,   # filter publish dt no earlier than
            to_dt: Optional[str] = None,   # filter publish dt no late than
            field: Optional[List[str]] = None,  # list of field of study
            min_citation_cnt: Optional[int] = 0,  # citation count no less than
            institutions: Optional[List[str]] = None,  # restrcted to list of institutions, to be implemented
            journals: Optional[List[str]] = None,  # restrcted to list of journals, to be implemented
            author_ids: Optional[List[str]] = None,  # restrcted to list of authors' ids      
            ):
        """get cited papers metadata"""
        # retrieve cited paper metadata from s2
        s2_citedpaper_metadata = self.s2.get_s2_cited_papers(paper_id, fields=field, limit=limit)
        
        # convert to standard format, be aware the output include nodes and edges 
        s2_citedpapermeta_json = process_citation_metadata(
            s2_citedpaper_metadata,
            from_dt,
            to_dt,
            field,
            min_citation_cnt,
            institutions,
            journals,
            author_ids)

        # iterate processed paper metadata, and store information to nodes_json and edge_json separately
        # nodes need to dedup, however, edges do not need to dedup
        node_id_pool = [x['id'] for x in self.nodes_json]
        for item in s2_citedpapermeta_json:

            if item['type'] == 'node':
                curr_node_id = item['id']
                if curr_node_id not in node_id_pool:  # for new node
                    item['properties']['source'] = ['CitedPaper']
                    item['properties']['sourceDesc'] = [f'cited by {paper_id}']
                    self.nodes_json.append(item)
                else:   # for existing node
                    idx = node_id_pool.index(curr_node_id)
                    if type(self.nodes_json[idx].get('source')) == list:
                        self.nodes_json[idx]['source'].append('CitedPaper')
                    else:
                        self.nodes_json[idx]['source'] = ['CitedPaper']
                    if type(self.nodes_json[idx].get('sourceDesc')) == list:
                        self.nodes_json[idx]['sourceDesc'].append(f'cited by {paper_id}')
                    else:
                        self.nodes_json[idx]['sourceDesc'] = [f'cited by {paper_id}']
            
            elif item['type'] == 'relationship':
                # we aim to construct multigraph, which allows multiple edges between any pair of nodes
                self.edges_json.append(item)


    def get_citing_papers(
            self, 
            paper_id, 
            limit:Optional[int]=100,
            from_dt: Optional[str] = None,   # filter publish dt no earlier than
            to_dt: Optional[str] = None,   # filter publish dt no late than
            field: Optional[List[str]] = None,  # list of field of study
            min_citation_cnt: Optional[int] = 0,  # citation count no less than
            institutions: Optional[List[str]] = None,  # restrcted to list of institutions, to be implemented
            journals: Optional[List[str]] = None,  # restrcted to list of journals, to be implemented
            author_ids: Optional[List[str]] = None,  # restrcted to list of authors' ids      
            ):
        """retrieve citing papers"""
        # retrieve citing paper metadata from s2
        s2_citingpaper_metadata = self.s2.get_s2_citing_papers(paper_id, fields=field, limit=limit)

        # convert to standard format, be aware the output include nodes and edges 
        s2_citingpaper_metadata = process_citation_metadata(
            s2_citingpaper_metadata,
            from_dt,
            to_dt,
            field,
            min_citation_cnt,
            institutions,
            journals,
            author_ids)

        # iterate processed paper metadata, and store information to nodes_json and edge_json separately
        # nodes need to dedup, however, edges do not need to dedup
        node_id_pool = [x['id'] for x in self.nodes_json]
        for item in s2_citingpaper_metadata:

            if item['type'] == 'node':
                curr_node_id = item['id']
                if curr_node_id not in node_id_pool:  # for new node
                    item['properties']['source'] = ['CitingPaper']
                    item['properties']['sourceDesc'] = [f"citing {paper_id}"]
                    self.nodes_json.append(item)
                else:   # for existing node
                    idx = node_id_pool.index(curr_node_id)
                    if type(self.nodes_json[idx].get('source')) == list:
                        self.nodes_json[idx]['source'].append('CitingPaper')
                    else:
                        self.nodes_json[idx]['source'] = ['CitingPaper']
                    if type(self.nodes_json[idx].get('sourceDesc')) == list:
                        self.nodes_json[idx]['sourceDesc'].append(f"citing {paper_id}")
                    else:
                        self.nodes_json[idx]['sourceDesc'] = [f"citing {paper_id}"]
            
            elif item['type'] == 'relationship':
                # we aim to construct multigraph, which allows multiple edges between any pair of nodes
                self.edges_json.append(item)


    def get_related_papers(
            self, 
            domain,
            input_text, 
            limit:Optional[int]=100,
            from_dt: Optional[str] = None,   # filter publish dt no earlier than
            to_dt: Optional[str] = None,   # filter publish dt no late than
            field: Optional[List[str]] = None,  # list of field of study
            min_citation_cnt: Optional[int] = 0,  # citation count no less than
            institutions: Optional[List[str]] = None,  # restrcted to list of institutions, to be implemented
            journals: Optional[List[str]] = None,  # restrcted to list of journals, to be implemented
            author_ids: Optional[List[str]] = None,  # restrcted to list of authors' ids      
            ):
        """further expand paper scope by leveraging deep search with LLMs"""
        # llm propose search queries
        keywords_topics_info = search_query_gen(domain, input_text, self.llm_api_key, self.llm_model_name)
        # extract keywords, topics, queries
        keywords_topics_json = json.loads(repair_json(keywords_topics_info))
        field_of_study = keywords_topics_json.get('field_of_study')
        keywords_and_topics = keywords_topics_json.get('keywords_and_topics')
        tags = keywords_topics_json.get('tags')
        queries = keywords_topics_json.get('queries')

        for query in queries:
            s2_paper_metadata = self.s2.search_paper_by_keywords(query, fields=field, limit=limit)

            # convert to standard format, be aware the output include nodes and edges 
            s2_papermeta_json = process_related_metadata(
                s2_paper_metadata,
                from_dt,
                to_dt,
                field,
                min_citation_cnt,
                institutions,
                journals,
                author_ids)

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
                        if type(self.nodes_json[idx].get('source')) == list:
                            self.nodes_json[idx]['source'].append('LLMQuery')
                        else:
                            self.nodes_json[idx]['source'] = ['LLMQuery']
                        if type(self.nodes_json[idx].get('sourceDesc')) == list:
                            self.nodes_json[idx]['sourceDesc'].append(query)
                        else:
                            self.nodes_json[idx]['sourceDesc'] = [query]
                
                elif item['type'] == 'relationship':
                    # we aim to construct multigraph, which allows multiple edges between any pair of nodes
                    self.edges_json.append(item)


    async def add_semantic_relationship(self, paper_nodes_json):
        """add semantic similarity as edges between paper node"""
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

        # then calculate semantic similarities between the texts
        embeds = await texts_embed_gen(texts, self.embed_api_key, self.embed_model_name)

        # calculate similarity matrix
        sim_matrix = semantic_similarity_gen(embeds, embeds)

        # iterate similarity matrix to add similarity relationships
        rows, cols = sim_matrix.shape
        for i in range(rows):
            for j in range(cols):
                sim = sim_matrix[i, j]
                if i != j:
                    # if edge exist, then update weith
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
