import asyncio
from typing import List, Dict, Optional, Union # Added Tuple

import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from collect.paper_query import PaperQuery
from collect.related_topic_query import TopicQuery
from collect.citation_query import CitationQuery
from collect.author_query import AuthorQuery
from collect.paper_recommendation import PaperRecommendation

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PaperSearch:
    """
    A class for exploring academic papers, optimized for asynchronous operations.
    """
    def __init__(
            self,
            # seed papers
            seed_research_topics: Optional[List] = None,
            seed_paper_titles: Optional[Union[List[str], str]] = None,
            seed_paper_dois: Optional[Union[List[str], str]] = None,
            # parameters
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
            nodes_json: Optional[List[Dict]] = None,
            edges_json: Optional[List[Dict]] = None, 
            search_limit: Optional[int] = 50,
            recommend_limit: Optional[int] = 50,
            citation_limit: Optional[int] = 100,
            # instances
            pq_instance: Optional[PaperQuery] = None, 
            aq_instance: Optional[AuthorQuery] = None, 
            cq_instance: Optional[CitationQuery] = None, 
            rq_instance: Optional[PaperRecommendation] = None, 
            tq_instance: Optional[TopicQuery] = None
    ):
        """Initialize PaperSearch parameters."""
        # seed papers info
        self.research_topics = seed_research_topics
        self.seed_paper_titles = [seed_paper_titles] if isinstance(seed_paper_titles, str) and seed_paper_titles else seed_paper_titles if isinstance(seed_paper_titles, list) else []
        self.seed_paper_dois = [seed_paper_dois] if isinstance(seed_paper_dois, str) and seed_paper_dois else seed_paper_dois if isinstance(seed_paper_dois, list) else []

        self.search_limit = search_limit
        self.recommend_limit = recommend_limit
        self.citation_limit = citation_limit

        # Filters
        self.from_dt = from_dt
        self.to_dt = to_dt
        self.fields_of_study = fields_of_study

        # State: Nodes and Edges
        self.nodes_json = []
        self._node_ids = set() # 维护为实例属性
        if isinstance(nodes_json, list) and len(nodes_json) > 0:
            self._add_items_json(nodes_json, source='initial')
            
        self.edges_json = []
        self._edge_tuples = set()
        if isinstance(edges_json, list) and len(edges_json) > 0:
            self._add_items_json(edges_json, source='initial')

        # store paper abstraction embeddings
        self.abs_embed_ref = {}

        # explored nodes
        self.explored_nodes = {'topic':set(),   # topic explored
                               'author':set(),   # authors explored 
                               'reference':set(),  # papers with reference explored
                               'citing': set(),  # papers with citation explored
                               'title': set(),  # papers with title search
                               'paper':set(), # papers with doi search
                               'recommendation':set(),  # papers for recommendations, format like ([pos_paper_dois], [neg_paper_dois])
                                }
        
        # exclusion list
        self.excluded_nodes = {'non_existing': set(),
                               'out_of_scope': set(),
        }
        self.removed_nodes = []
        self.removed_edges = []


        # initiate instances
        self.pq = pq_instance if pq_instance else PaperQuery()
        self.tq = tq_instance if tq_instance else TopicQuery()
        self.aq = aq_instance if aq_instance else AuthorQuery()
        self.cq = cq_instance if cq_instance else CitationQuery()
        self.rq = rq_instance if rq_instance else PaperRecommendation()


    ############################################################################
    # basic functions
    ############################################################################
    def _add_items_json(
            self, 
            items: List[Dict], 
            source: Optional[str]='unknown'):
        """Adds nodes and relationships from processed data, remove duplications
        Args:
            items: node json or edge json
            round / source: indicating from which iteration the node or edge was created.
                            round and source help to track the dynamics of the data generation
        """
        items_to_add_node = []
        items_to_add_edge = []

        for item in items:
            if item['type'] == 'node':
                node_id = item['id']
                if node_id not in self._node_ids:
                    item['dataGeneration'] = {'source': source}
                    items_to_add_node.append(item)
                    if item['labels'][0] in ['Author', 'Paper']:
                        if item.get('properties', {}).get('is_complete', False) == True:
                            self._node_ids.add(item['id'])
                    else:
                        self._node_ids.add(item['id'])

            elif item['type'] == 'relationship':
                rel_type = item['relationshipType']
                start_id = item['startNodeId']
                end_id = item['endNodeId']
                edge_tuple = (start_id, end_id, rel_type) 
                if edge_tuple not in self._edge_tuples:
                    item['dataGeneration'] = {'source': source}
                    items_to_add_edge.append(item)
                    self._edge_tuples.add(edge_tuple)

        self.nodes_json.extend(items_to_add_node)
        self.edges_json.extend(items_to_add_edge)


    async def topic_generation(
            self,
            seed_paper_json: List[Dict],
            llm_api_key: str,
            llm_model_name: str
            ):
        """use LLM to generate related topics based on seed paper information"""
        logging.info("Use LLM to identify key related topics.")
        keywords_topics_json = await self.tq.llm_gen_related_topics(seed_paper_json, llm_api_key, llm_model_name)
        topics_json = self.tq.get_topic_json(keywords_topics_json, seed_paper_json)

        # add topic to json
        self._add_items_json(topics_json, source="topic_extention")

        return keywords_topics_json
 
    ############################################################################
    # paper collection functions
    ############################################################################
    async def paper_search(
            self,
            # for paper info
            paper_titles: Optional[List]=[],
            paper_dois: Optional[List]=[],
            # search params
            search_limit: Optional[int] = 50,
            from_dt: Optional[str]="2000-01-01",
            to_dt: Optional[str]="9999-12-31",
            fields_of_study: Optional[List] = None
        ):
        """basic search for seed paper related information"""
        paper_titles = [x for x in paper_titles if x not in self.explored_nodes['title']]
        paper_dois = [x for x in paper_dois if x not in self.explored_nodes['paper']]

        if len(paper_titles) > 0 or len(paper_dois) > 0:
            logging.info(f"Search {len(paper_titles)} paper titles and {len(paper_dois)} for paper information.")
            papers_json = await self.pq.get_paper_info(
                seed_paper_titles=paper_titles,
                seed_paper_dois=paper_dois,
                limit=search_limit, 
                from_dt=from_dt, 
                to_dt=to_dt,
                fields_of_study=fields_of_study)

            # add item as json
            self._add_items_json(papers_json, source='paper_search')
            logging.info(f"After paper search: Nodes: {len(self.nodes_json)}, Edges: {len(self.edges_json)}")

        # update explored nodes
        if len(paper_titles) > 0:
            self.explored_nodes['title'].update(paper_titles)
        if len(paper_dois) > 0:
            self.explored_nodes['paper'].update(paper_dois)


    async def topic_search(
            self,
            # for paper info
            topics: Optional[List]=[],
            # search params
            search_limit: Optional[int] = 50,
            from_dt: Optional[str]="2000-01-01",
            to_dt: Optional[str]="9999-12-31",
            fields_of_study: Optional[List] = None
        ):
        """basic search for seed paper related information"""
        topics = [x for x in topics if x not in self.explored_nodes['topic']]

        if len(topics) > 0:
            logging.info(f"Search {len(topics)} topics for paper information.")
            topics_json = await self.tq.get_topic_papers(
                topic_queries=topics,
                limit=search_limit, 
                from_dt=from_dt, 
                to_dt=to_dt,
                fields_of_study=fields_of_study)
        
            # add item as json
            self._add_items_json(topics_json, source='topic_search')
            logging.info(f"After paper search: Nodes: {len(self.nodes_json)}, Edges: {len(self.edges_json)}")

        # update explored nodes
        if len(topics) > 0:
            self.explored_nodes['topic'].update(topics)


    async def author_search(
            self,
            # for author info
            author_ids: Optional[List]=[],
            # search params
            from_dt: Optional[str]="2000-01-01",
            to_dt: Optional[str]="9999-12-31",
            fields_of_study: Optional[List] = None
        ):
        """basic search for author related information"""
        author_ids = [x for x in author_ids if x not in self.explored_nodes['author']]

        if len(author_ids) > 0:
            logging.info(f"Search {len(author_ids)} author information.")
            author_json = await self.aq.get_author_info(author_ids, from_dt, to_dt, fields_of_study)
        
            # add item as json
            self._add_items_json(author_json, source='author_search')
            logging.info(f"After author search: Nodes: {len(self.nodes_json)}, Edges: {len(self.edges_json)}")

        # update explored nodes
        if len(author_ids) > 0:
            self.explored_nodes['author'].update(author_ids)


    async def citation_search(
            self,
            # for citation info
            ref_paper_dois: Optional[List]=[],
            citing_paper_dois: Optional[List]=[],
            # search params
            citation_limit: Optional[int] = 100,
            from_dt: Optional[str]="2000-01-01",
            to_dt: Optional[str]="9999-12-31",
            fields_of_study: Optional[List] = None
        ):
        """citation search for reference / citing information"""
        ref_paper_dois = [x for x in ref_paper_dois if x not in self.explored_nodes['reference']]
        citing_paper_dois = [x for x in citing_paper_dois if x not in self.explored_nodes['citing']]

        # setup tasks
        logging.info(f"Preparing reference for {len(ref_paper_dois)} papers and citing for {len(citing_paper_dois)} papers.")
        tasks = []
        for paper_doi in ref_paper_dois:
            tasks.append(self.cq.get_cited_papers(paper_doi, citation_limit, from_dt, to_dt, fields_of_study))
        for paper_doi in citing_paper_dois:
            tasks.append(self.cq.get_citing_papers(paper_doi, citation_limit, from_dt, to_dt, fields_of_study))

        # run tasks
        citation_json = []
        if tasks:
            logging.info(f"Running {len(tasks)} citation collection tasks concurrently...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logging.error(f"Citation collection task failed: {result}")
                elif isinstance(result, list):
                    citation_json.extend(result) # Add items from successful tasks
                    logging.info("Main citation collection tasks finished.")
        else:
            logging.info("No main citation collection tasks to run.")

        # add item as json
        self._add_items_json(citation_json, source='citation_search')
        logging.info(f"After citation search: Nodes: {len(self.nodes_json)}, Edges: {len(self.edges_json)}")

        # update explored nodes
        if len(ref_paper_dois) > 0:
            self.explored_nodes['reference'].update(ref_paper_dois)
        if len(citing_paper_dois) > 0:
            self.explored_nodes['citing'].update(citing_paper_dois)


    async def paper_recommendation(
            self,
            # for S2 recommendations
            pos_paper_dois: Optional[List] = [],
            neg_paper_dois: Optional[List] = [],
            # search params
            recommend_limit: Optional[int] = 100,
            from_dt: Optional[str]="2000-01-01",
            to_dt: Optional[str]="9999-12-31",
            fields_of_study: Optional[List] = None
        ):
        """paper recommendations based on posiive paper dois and negtive paper dois"""
        if len(pos_paper_dois) > 0:
            logging.info(f"Recommend papers based on {len(pos_paper_dois)} positive papers and {len(neg_paper_dois)} papers.")
            recommend_json = await self.rq.get_recommend_papers(
                pos_paper_dois,
                neg_paper_dois,
                recommend_limit, 
                from_dt, 
                to_dt, 
                fields_of_study)

            # add item as json
            self._add_items_json(recommend_json, source='paper_recommendation')
            logging.info(f"After recommendation: Nodes: {len(self.nodes_json)}, Edges: {len(self.edges_json)}")

        # update explored nodes
        if len(pos_paper_dois) > 0:
            self.explored_nodes['recommendation'].add((tuple(sorted(pos_paper_dois)), tuple(sorted(neg_paper_dois))))


    async def consolidated_search(
            self,
            # for paper / author info
            topics: Optional[List]=[],
            paper_titles: Optional[List]=[],
            paper_dois: Optional[List]=[],
            author_ids: Optional[List]=[],
            # for citation info
            ref_paper_dois: Optional[List]=[],
            citing_paper_dois: Optional[List]=[],
            # for S2 recommendations
            pos_paper_dois: Optional[List] = [],
            neg_paper_dois: Optional[List] = [],
            # search params
            search_limit: Optional[int] = 50,
            citation_limit: Optional[int] = 100,
            recommend_limit: Optional[int] = 100,
            from_dt: Optional[str]="2000-01-01",
            to_dt: Optional[str]="9999-12-31",
            fields_of_study: Optional[List] = None
        ):
        """consolidate paper search, author search, citation search and paper recommendations"""
        tasks = []

        if len(topics) > 0:
            tasks.append(self.topic_search(
                topics,
                search_limit,
                from_dt,
                to_dt,
                fields_of_study))         

        if len(paper_titles) > 0 or len(paper_dois) > 0:
            tasks.append(self.paper_search(
                paper_titles,
                paper_dois,
                search_limit,
                from_dt,
                to_dt,
                fields_of_study))
        
        if len(author_ids) > 0:
            tasks.append(self.author_search(
                author_ids,
                from_dt,
                to_dt,
                fields_of_study))
        
        if len(ref_paper_dois) > 0 or len(citing_paper_dois) > 0:
            tasks.append(self.citation_search(
                ref_paper_dois,
                citing_paper_dois,
                citation_limit,
                from_dt,
                to_dt,
                fields_of_study))

        if len(pos_paper_dois) > 0:
            tasks.append(self.paper_recommendation(
                pos_paper_dois,
                neg_paper_dois,
                recommend_limit,
                from_dt,
                to_dt,
                fields_of_study))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            if len(results) > 0:
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        # help track error information
                        logging.error(f"A sub-task in consolidated_search failed: {result}", exc_info=True)

