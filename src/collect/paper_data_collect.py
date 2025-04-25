import asyncio
from typing import List, Dict, Optional, Union, Any

from semanticscholar.Paper import Paper
from semanticscholar.Author import Author
from semanticscholar.Reference import Reference

import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from apis.s2_api import SemanticScholarKit
from collect.paper_data_process import (process_papers_data, process_authors_data, 
                                        process_citations_data, process_topics_data)

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
            seed_paper_ids: Optional[Union[List[str], str]] = None,
            # parameters
            from_dt: Optional[str] = None,
            to_dt: Optional[str] = None,
            fields_of_study: Optional[List[str]] = None,
            nodes_json: Optional[List[Dict]] = None,
            edges_json: Optional[List[Dict]] = None, 
            author_paper_limit: Optional[int] = 10,
            search_limit: Optional[int] = 50,
            recommend_limit: Optional[int] = 50,
            citation_limit: Optional[int] = 100,
            # instances
            s2_instance: Optional[SemanticScholarKit] = None, 
    ):
        """Initialize PaperSearch parameters.
        
        """
        # seed papers info
        self.research_topics = seed_research_topics
        self.seed_paper_titles = [seed_paper_titles] if isinstance(seed_paper_titles, str) and seed_paper_titles else seed_paper_titles if isinstance(seed_paper_titles, list) else []
        self.seed_paper_ids = [seed_paper_ids] if isinstance(seed_paper_ids, str) and seed_paper_ids else seed_paper_ids if isinstance(seed_paper_ids, list) else []

        self.author_paper_limit = author_paper_limit
        self.search_limit = search_limit
        self.recommend_limit = recommend_limit
        self.citation_limit = citation_limit

        # Filters
        self.from_dt = from_dt
        self.to_dt = to_dt
        self.fields_of_study = fields_of_study

        # State: Nodes and Edges
        self.nodes_json: List[Dict] = list(nodes_json) if nodes_json else []
        self.edges_json: List[Dict] = list(edges_json) if edges_json else []

        # store paper abstraction embeddings
        self.abs_embed_ref = {}

        # explored nodes
        self.explored_nodes = {'topic':set(),   
                               'author':set(),  
                               'reference':set(),  
                               'citing': set(),  
                               'title': set(),  
                               'paper':set(), 
                               'recommendation':set(),  # papers for recommendations, format like ([pos_paper_ids], [neg_paper_ids])
                               'paper_author': set(),  # search for all authors of a paper  
                                }
        
        self.not_found_nodes = {'topic':set(), 
                               'author':set(), 
                               'reference':set(), 
                               'citing': set(), 
                               'title': set(), 
                               'paper':set(), 
                               'recommendation':set(),  
                               'paper_author': set(),  
                                }
        
        self.data_pool = {'paper': [],
                          'author':[],
                          'reference': [],
                          'citing': [],
                          'topic': []
                         }

        # initiate instances
        self.s2 = s2_instance if s2_instance and isinstance(s2_instance, SemanticScholarKit) else SemanticScholarKit()

 
    ############################################################################
    # paper collection functions
    ############################################################################
    async def paper_search(
            self,
            paper_titles: Optional[List]=[],
            paper_ids: Optional[List]=[]  # Expects dois, arxiv ids, s2 paper ids
        ):
        """
        Asynchronously searches for papers by titles and various IDs (S2 handles resolution).
        Adds found raw paper metadata to self.data_pool['paper'].
        """
        paper_titles = paper_titles or []
        paper_ids = paper_ids or []

        # --- 1. Filter out already explored items ---
        titles_to_search = [t for t in paper_titles if t and t not in self.explored_nodes['title']]
        ids_to_search = [pid for pid in paper_ids if pid and pid not in self.explored_nodes['paper']]

        if not titles_to_search and not ids_to_search:
            logging.info("paper_search: No new titles or IDs to search.")
            return
    
        logging.info(f"Search {len(paper_titles)} paper titles and {len(paper_ids)} for paper information.")
        
        # --- 2. Create search tasks ---
        tasks_with_source = []
        task_coroutines = []

        # Task for searching by IDs (S2 get_papers handles various ID types)
        if ids_to_search:
            logging.info(f"paper_search: Creating task for {len(ids_to_search)} IDs...")
            coro = self.s2.get_papers(paper_ids=ids_to_search) # S2 handles resolution
            tasks_with_source.append({'source': 'id', 'values': ids_to_search, 'result': coro})
            task_coroutines.append(coro)

        # Tasks for searching by titles (one task per title)
        if titles_to_search:
            logging.info(f"paper_search: Creating {len(titles_to_search)} tasks for titles...")
            for title in titles_to_search:
                coro = self.s2.search_paper(query=title, limit=5, fields=['paperId', 'title']) # Limit results for title match
                tasks_with_source.append({'source': 'title', 'values': [title], 'result': coro})
                task_coroutines.append(coro)

        # --- 3. Execute tasks and collect results ---
        if task_coroutines:
            logging.info(f"paper_search: Running {len(task_coroutines)} query tasks concurrently...")
            results = await asyncio.gather(*task_coroutines, return_exceptions=True)

            if results:
                for i, task_info in enumerate(tasks_with_source):
                    source = task_info['source']
                    values = task_info['values'] # List of IDs or single title list
                    result = results[i]

                    if isinstance(result, Exception):
                        logging.error(f"paper_search: Task for source '{source}' ({values}) failed: {result}", exc_info=False) # Reduce noise
                        self.not_found_nodes[source].update(values)

                    elif result is not None:
                        if isinstance(result, list):  # search by ids
                            papers_dict = [item.raw_data for item in result if isinstance(item, Paper)]
                            self.data_pool['paper'].extend(papers_dict)
                        elif isinstance(result, Paper): 
                            self.data_pool['paper'].add(result.raw_data)
                        
                    else:
                        # Handle cases where the API might return None without an exception
                        logging.warning(f"Task for source '{source}' ({values}) returned None.")
                        if source == 'id':
                            self.not_found_nodes['paper'].update(values)
                        else:
                            self.not_found_nodes['title'].update(values)
            else:
                logging.warning("No paper query criteria (ID, Title) provided.")

        # --- 4. Update global status ---
        # update explored nodes
        if len(paper_titles) > 0:
            self.explored_nodes['title'].update(paper_titles)
        if len(paper_ids) > 0:
            self.explored_nodes['paper'].update(paper_ids)


    async def topic_search(
            self,
            topics: Optional[List]=[],
            search_limit: Optional[int] = 50,
            from_dt: Optional[str] = "2000-01-01",
            to_dt: Optional[str] = "9999-12-31",
            fields_of_study: Optional[List] = None
        ):
        """
        Asynchronously searches for papers by topic.
        Adds found raw paper metadata and topic links to the data pool.
        """
        topics = topics or []
        search_limit = search_limit if search_limit is not None else self.search_limit
        from_dt = from_dt or self.from_dt
        to_dt = to_dt or self.to_dt
        fields_of_study = fields_of_study or self.fields_of_study

        # --- 1. Filter ---
        topics_to_search = [t for t in topics if t and t not in self.explored_nodes['topic']]
        if not topics_to_search:
            logging.info("topic_search: No new topics to search.")
            return

        logging.info(f"topic_search: Searching {len(topics_to_search)} topics.")

        # --- 2. Create tasks ---
        tasks = []
        publication_date_filter = None
        if from_dt or to_dt:
             start = from_dt if from_dt else "*"
             end = to_dt if to_dt else "*"
             publication_date_filter = f"{start}:{end}"

        for topic in topics_to_search:
            tasks.append(self.s2.search_paper(
                query=topic,
                limit=search_limit,
                publication_date_or_year=publication_date_filter,
                fields_of_study=fields_of_study
            ))

        # --- 3. Execute and collect ---
        if tasks:
            logging.info(f"topic_search: Running {len(tasks)} topic search tasks concurrently...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            new_papers_count = 0
            new_topic_links = 0
            existing_paper_ids = {p.get('paperId') for p in self.data_pool['paper'] if p.get('paperId')}

            for idx, result in enumerate(results):
                current_topic = topics_to_search[idx]
                if isinstance(result, Exception):
                    logging.error(f"topic_search: Task for topic '{current_topic}' failed: {result}", exc_info=False)
                    self.not_found_nodes['topic'].add(current_topic)

                elif hasattr(result, '_items') and result._items:
                    papers_list = result._items
                    # Add paper metadata to data pool (deduplicated)
                    papers_dict = [item.raw_data for item in papers_list if isinstance(item, Paper) and hasattr(item, 'raw_data')]
                    new_papers = [p for p in papers_dict if p.get('paperId') and p['paperId'] not in existing_paper_ids]
                    self.data_pool['paper'].extend(new_papers)
                    new_papers_count += len(new_papers)
                    existing_paper_ids.update(p['paperId'] for p in new_papers) # Update seen IDs

                    # Add topic metadata links
                    paper_ids_in_result = [paper.get('paperId') for paper in papers_dict if paper.get('paperId')]
                    for pid in paper_ids_in_result:
                        # Use 'paperId' key for consistency
                        self.data_pool['topic'].append({'topic': current_topic, 'paperId': pid})
                        new_topic_links += 1
                else:
                    logging.warning(f"topic_search: Task for topic '{current_topic}' returned no results or failed silently.")
                    self.not_found_nodes['topic'].add(current_topic) # Mark as not found if no results

            logging.info(f"topic_search: Added {new_papers_count} new papers and {new_topic_links} topic links to data pool.")

        # --- 4. Update status ---
        if len(topics) > 0:
            self.explored_nodes['topic'].update(topics)


    async def authors_search(
            self,
            author_ids: Optional[List]=[],
        ):
        """
        Asynchronously fetches author details and their associated papers.
        Adds raw author and paper metadata to the data pool.
        """
        author_ids = author_ids or []
        # --- 1. Filter ---
        ids_to_search = [aid for aid in author_ids if aid and aid not in self.explored_nodes['author']]
        if not ids_to_search:
            logging.info("authors_search: No new author IDs to search.")
            return

        logging.info(f"authors_search: Searching {len(ids_to_search)} authors.")

        # --- 2. Conduct search (one call for multiple authors) ---
        # return a list of authors (with papers in each author)
        authors_papers_metadata = await self.s2.get_authors(author_ids=ids_to_search)
    
        # --- 3. Process results and update pool ---
        new_papers_count = 0
        new_authors_count = 0
        if authors_papers_metadata:
            existing_paper_ids = {p.get('paperId') for p in self.data_pool['paper'] if p.get('paperId')}
            existing_author_ids = {a.get('authorId') for a in self.data_pool['author'] if a.get('authorId')}

            for author_item in authors_papers_metadata:
                if not isinstance(author_item, Author) or not hasattr(author_item, 'raw_data'):
                    continue

                current_author_id = author_item.authorId
                if not current_author_id: continue

                # Process papers associated with the author
                papers_dict = []
                if hasattr(author_item, 'papers') and author_item.papers:
                    papers_dict = [paper_item.raw_data for paper_item in author_item.papers if isinstance(paper_item, Paper) and hasattr(paper_item, 'raw_data')]

                new_papers = [p for p in papers_dict if p.get('paperId') and p['paperId'] not in existing_paper_ids]
                self.data_pool['paper'].extend(new_papers)
                new_papers_count += len(new_papers)
                existing_paper_ids.update(p['paperId'] for p in new_papers)

                # Process author data (excluding papers field)
                if current_author_id not in existing_author_ids:
                    author_dict = {
                        k: v for k, v in author_item.raw_data.items()
                        if k != 'papers' and v is not None # Exclude papers field
                    }
                    self.data_pool['author'].append(author_dict)
                    new_authors_count += 1
                    existing_author_ids.add(current_author_id)

                # Mark this author ID as explored successfully
                self.explored_nodes['author'].add(current_author_id)
            logging.info(f"authors_search: Added {new_authors_count} new authors and {new_papers_count} new papers to data pool.")

        # --- 4. Update status ---
        # Update explored nodes for those initially requested but not found in results 
        found_author_ids = {a.authorId for a in authors_papers_metadata if isinstance(a, Author) and a.authorId}
        not_found_in_results = set(ids_to_search) - found_author_ids
        self.not_found_nodes['author'].update(not_found_in_results)
        self.explored_nodes['author'].update(ids_to_search)


    async def paper_author_search(
            self,
            paper_ids: List[str],
            limit: int = 100,  # restrict # of authors
            ):
        """
        Asynchronously fetches authors for specific papers (details only, no papers).
        Adds raw author metadata to the data pool.
        """
        limit = limit if limit is not None else self.author_paper_limit
        # --- 1. Filter ---
        ids_to_search = [pid for pid in paper_ids if pid and pid not in self.explored_nodes['paper_author']]
        if not ids_to_search:
            logging.info("paper_author_search: No new paper IDs to search for authors.")
            return
        
        logging.info(f"paper_author_search: Searching authors for {len(ids_to_search)} papers.")
        
        # --- 2. Create tasks ---
        tasks = []
        for pid in ids_to_search:
            # Fetches authors for a single paper
            tasks.append(self.s2.get_paper_authors(paper_id=pid, limit=limit))
        
        # --- 3. Execute and collect ---
        all_authors_metadata: List[Any] = []
        if tasks:
            logging.info(f"paper_author_search: Running {len(tasks)} author search tasks concurrently...")
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for idx, result in enumerate(results):
                current_paper_id = ids_to_search[idx]
                if isinstance(result, Exception):
                    logging.error(f"paper_author_search: Task for paper '{current_paper_id}' failed: {result}", exc_info=False)
                    self.not_found_nodes['paper_author'].add(current_paper_id)
                
                elif hasattr(result, '_items') and result._items: # Result is PaginatedResult
                    all_authors_metadata.extend(result._items)
                
                elif isinstance(result, list): # Should ideally be PaginatedResult, but handle list just in case
                     all_authors_metadata.extend(result)
                else:
                    logging.warning(f"paper_author_search: Task for paper '{current_paper_id}' returned no results or failed silently.")
                    self.not_found_nodes['paper_author'].add(current_paper_id) # Mark as not found


        # --- 3. Collect results data ---
        # Run all query tasks concurrently
        if tasks:
            logging.info(f"Running {len(tasks)} author search for given papers query tasks concurrently...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    logging.error(f"Related paper search task failed: {result}")
                    self.not_found_nodes['paper_author'] = paper_ids[idx]
                elif isinstance(result, list):
                    authors_metadata.extend(result)
                else:
                    logging.warning(f"Task for paper '{paper_ids[idx]}' returned None.")
                    self.not_found_nodes['paper_author'].add(paper_ids[idx])

    # --- 4. Update global status ---
    # update explored nodes
    if len(paper_ids) > 0:
        self.explored_nodes['paper_author'].update(paper_ids)
    
    # split authors and papers
    if len(authors_metadata) > 0:
        for author_item in authors_metadata:
            if not isinstance(author_item, Author):
                continue
            # drop papers in author
            author_dict = {
                k: v for k, v in author_item.raw_data.items()
                if k not in ['papers'] and v is not None
            }
            self.data_pool['author'].append(author_dict)


    async def reference_search(
            self,
            ref_paper_ids: List[str],
            citation_limit: Optional[int] = 100,
        ):
        """citation search for reference information"""
        # --- 1. Process paper ids to avoid duplicated search ---
        ref_paper_ids = [x for x in ref_paper_ids if x not in self.explored_nodes['reference']]

        # --- 2. Conduct search via Semantic Scholar ---
        if len(ref_paper_ids) > 0:
            tasks = []
            logging.info(f"Preparing reference for {len(ref_paper_ids)} papers ...")
            for pid in ref_paper_ids:
                # return a list of references
                tasks.append(self.s2.get_paper_references(paper_id=pid, limit=citation_limit))
                
            # --- 3. Collect results data ---
            # Run all query tasks concurrently
            if tasks:
                logging.info(f"Running {len(tasks)} reference search tasks concurrently...")
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for idx, result in enumerate(results):
                    if isinstance(result, Exception):
                        logging.error(f"Related paper search task failed: {result}")
                        self.not_found_nodes['reference'] = ref_paper_ids[idx]
                    elif isinstance(result, list) and len(result) > 0:
                        for ref_item in result:
                            if not isinstance(ref_item, Reference):
                                continue
                            # get paper
                            paper = ref_item.citedPaper
                            if isinstance(paper, Paper):
                                self.data_pool['paper'].append(paper.raw_data)
                                # drop papers in reference
                                ref_dict = {
                                    k: v for k, v in ref_item.raw_data.items()
                                    if k not in ['citedPaper'] and v is not None
                                }
                                ref_dict['source_id'] = ref_paper_ids[idx]
                                ref_dict['target_id'] = paper.paperId 
                                ref_dict['citation_type'] = 'CITES'
                                self.data_pool['reference'].append(ref_dict)
                    else:
                        logging.warning(f"Task for paper '{ref_paper_ids[idx]}' returned None.")
                        self.not_found_nodes['reference'].add(ref_paper_ids[idx])

        # --- 4. Update global status ---
        # update explored nodes
        if len(ref_paper_ids) > 0:
            self.explored_nodes['reference'].update(ref_paper_ids)


    async def citing_search(
            self,
            cit_paper_ids: Optional[List]=[],
            citation_limit: Optional[int] = 100,
        ):
        """citation search for reference information"""
        # --- 1. Process paper ids to avoid duplicated search ---
        cit_paper_ids = [x for x in cit_paper_ids if x not in self.explored_nodes['citing']]

        # --- 2. Conduct search via Semantic Scholar ---
        if len(cit_paper_ids) > 0:
            tasks = []
            logging.info(f"Preparing citing for {len(cit_paper_ids)} papers ...")
            for pid in cit_paper_ids:
                # return a list of references
                tasks.append(self.s2.get_paper_citations(paper_id=pid, limit=citation_limit))
                
            # --- 3. Collect results data ---
            # Run all query tasks concurrently
            if tasks:
                logging.info(f"Running {len(tasks)} citing search tasks concurrently...")
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for idx, result in enumerate(results):
                    if isinstance(result, Exception):
                        logging.error(f"Related paper search task failed: {result}")
                        self.not_found_nodes['citing'] = cit_paper_ids[idx]
                    elif isinstance(result, list):
                        for cit_item in result:
                            if not isinstance(cit_item, Reference):
                                continue
                            # get paper
                            paper = cit_item.citedPaper
                            if isinstance(paper, Paper):
                                self.data_pool['paper'].append(paper.raw_data)
                            # drop papers in citing
                            cit_dict = {
                                k: v for k, v in cit_item.raw_data.items()
                                if k not in ['citingPaper'] and v is not None
                            }
                            cit_dict['source_id'] = paper.paperId 
                            cit_dict['target_id'] = cit_paper_ids[idx]
                            cit_dict['citation_type'] = 'CITES'
                            self.data_pool['citing'].append((cit_paper_ids[idx], 'citingPaper', cit_dict))
                    else:
                        logging.warning(f"Task for paper '{cit_paper_ids[idx]}' returned None.")
                        self.not_found_nodes['citing'].add(cit_paper_ids[idx])

        # --- 4. Update global status ---
        # update explored nodes
        if len(cit_paper_ids) > 0:
            self.explored_nodes['citing'].update(cit_paper_ids)


    async def paper_recommendation(
            self,
            pos_paper_ids: List[str],
            neg_paper_ids: Optional[List] = [],
            recommend_limit: Optional[int] = 100,
        ):
        """paper recommendations based on posiive paper ids and negtive paper ids"""
        # --- 1. Get recommend papers from Semantic Scholar ---
        if len(pos_paper_ids) > 0:
            logging.info(f"Recommend papers based on {len(pos_paper_ids)} positive papers and {len(neg_paper_ids)} papers.")
            papers_metadata = await self.s2.get_recommended_papers_from_lists(
                positive_paper_ids=pos_paper_ids,
                negative_paper_ids=neg_paper_ids,
                limit=recommend_limit)

        # --- 3. Update global status ---
        # update explored nodes
        if len(pos_paper_ids) > 0:
            self.explored_nodes['recommendation'].add((tuple(sorted(pos_paper_ids)), tuple(sorted(neg_paper_ids))))

        # add paper metadata to data pool
        if len(papers_metadata) > 0:
            papers_dict = [item.raw_data for item in papers_metadata if isinstance(item, Paper)]
            self.data_pool['paper'].extend(papers_dict)


    ############################################################################
    # paper consolidated search
    ############################################################################
    async def consolidated_search(
            self,
            # for paper / author info
            topics: Optional[List]=[],
            paper_titles: Optional[List]=[],
            paper_ids: Optional[List]=[],
            author_ids: Optional[List]=[],
            # for citation info
            ref_paper_ids: Optional[List]=[],
            citing_paper_ids: Optional[List]=[],
            # for S2 recommendations
            pos_paper_ids: Optional[List] = [],
            neg_paper_ids: Optional[List] = [],
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

        if len(paper_titles) > 0 or len(paper_ids) > 0:
            tasks.append(self.paper_search(paper_titles, paper_ids))
        
        if len(author_ids) > 0:
            tasks.append(self.authors_search(author_ids))
        
        if len(ref_paper_ids) > 0:
            tasks.append(self.reference_search(ref_paper_ids, citation_limit))

        if len(citing_paper_ids) > 0:
            tasks.append(self.citing_search(citing_paper_ids, citation_limit))

        if len(pos_paper_ids) > 0:
            tasks.append(self.paper_recommendation(pos_paper_ids, neg_paper_ids, recommend_limit))

        if len(topics) > 0:
            tasks.append(self.topic_search(topics, search_limit, from_dt, to_dt, fields_of_study))  

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            if len(results) > 0:
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        # help track error information
                        logging.error(f"A sub-task in consolidated_search failed: {result}", exc_info=True)


    ############################################################################
    # paper search results post processing
    ############################################################################
    async def supplement_abstract(
            self, 
            paper_ids:Union[List[str], str]):
        """search from Semantic Scholar for paper abstract information"""
        if isinstance(paper_ids, str):
            paper_ids = [paper_ids]

        if len(paper_ids) > 0:
            logging.info(f"Fetching {len(paper_ids)} papers for abstracts ...")
            papers_abstract_info = await self.s2.get_papers(paper_ids=paper_ids, fields=['paperId', 'abstract'])
            papers_abstract_ref = {item['paperId']:item['abstract'] for item in papers_abstract_info}
            return papers_abstract_ref
        else:
            return {}

    async def post_process(
            self,
            if_supplement_abstract: Optional[bool] = True,  # whether add abstract information
            ):
        """post process collected paper metadata"""
        nodes_edges_json = []

        if len(self.data_pool['paper']) > 0:
            papers_json = process_papers_data(self.data_pool['paper'])

            # supplement abstract for papers
            if if_supplement_abstract:
                null_abstract_ids = []
                for item in papers_json:
                    if (item.get('type') == 'node' and item.get('label') == ['Paper'] 
                        and item.get('properties', {}).get('abstract') is None):
                        null_abstract_ids.append(item['id'])

                papers_abstract_ref = await self.supplement_abstract(null_abstract_ids)
                for item in papers_json:
                    if item.get('type') == 'node' and item.get('label') == ['Paper']:
                        pid = item['id']
                        abstract = papers_abstract_ref.get(pid)
                        if abstract is not None:
                            item['properties']['abstract'] = abstract
            
            nodes_edges_json.extend(papers_json)

        if len(self.data_pool['author']) > 0:
            authors_json = process_authors_data(self.data_pool['author'])
            nodes_edges_json.extend(authors_json)

        if len(self.data_pool['topic']) > 0:
            topics_json = process_topics_data(self.data_pool['topic'])
            nodes_edges_json.extend(topics_json)

        if len(self.data_pool['reference']) > 0:
            references_json = process_citations_data(self.data_pool['reference'])
            nodes_edges_json.extend(references_json)

        if len(self.data_pool['citing']) > 0:
            citings_json = process_citations_data(self.data_pool['citing'])
            nodes_edges_json.extend(citings_json)

        # allow duplication in nodes and edges, would dedup in later stage
        for item_json in nodes_edges_json:
            if item_json.get('type') == 'node':
                self.nodes_json.append(item_json)
            elif item_json.get('type') == 'relationship':
                self.edges_json.append(item_json)
