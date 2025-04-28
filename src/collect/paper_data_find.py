# paper_data_find.py
# utilize semantic scholar to collect and process paper related metadata

import re
import asyncio
from typing import List, Dict, Optional, Union, Any, Set, Tuple

from semanticscholar.Paper import Paper
from semanticscholar.Author import Author
from semanticscholar.Citation import Citation
from semanticscholar.Reference import Reference
from semanticscholar.PaginatedResults import PaginatedResults

import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from apis.s2_api import SemanticScholarKit

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


NODE_PAPER = "Paper"
NODE_AUTHOR = "Author"


class PaperFinder:
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
            author_paper_limit: Optional[int] = 10,
            search_limit: Optional[int] = 50,
            recommend_limit: Optional[int] = 50,
            citation_limit: Optional[int] = 100,
            # instances
            s2_instance: Optional[SemanticScholarKit] = None, 
    ):
        """Initialize PaperFinder parameters."""
        # ------- SEED PAPER INFORMATION ------
        # seed papers info
        self.research_topics = seed_research_topics
        self.seed_paper_titles = [seed_paper_titles] if isinstance(seed_paper_titles, str) and seed_paper_titles else seed_paper_titles if isinstance(seed_paper_titles, list) else []
        self.seed_paper_ids = [seed_paper_ids] if isinstance(seed_paper_ids, str) and seed_paper_ids else seed_paper_ids if isinstance(seed_paper_ids, list) else []

        # ------- SEARCH RELATED PARAMETERS ------
        # max count in search
        self.author_paper_limit = author_paper_limit
        self.search_limit = search_limit
        self.recommend_limit = recommend_limit
        self.citation_limit = citation_limit

        # Filters
        self.from_dt = from_dt
        self.to_dt = to_dt
        self.fields_of_study = fields_of_study

        # ------- GLOBAL STATUS VARIABLES ------
        # initiate semantic scholar instances
        self.s2 = s2_instance if s2_instance and isinstance(s2_instance, SemanticScholarKit) else SemanticScholarKit()

        # store id references, from doi id to doi
        self.paper_id_ref = {}  # k-v dict where key is doi and value is s2 paper id (also unique id here)

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
        
        # not found node (due to non-existing or errors)
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
 
    ############################################################################
    # paper search relatied functions
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
                coro = self.s2.search_paper(query=title, limit=5, match_title=True) # Limit results for title match
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
                            self.data_pool['paper'].append(result.raw_data)
                        
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
        if titles_to_search:
            self.explored_nodes['title'].update(titles_to_search)
        if ids_to_search:
            self.explored_nodes['paper'].update(ids_to_search)


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

                elif isinstance(result, PaginatedResults) and hasattr(result, '_items') and result._items:
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
        self.explored_nodes['topic'].update(topics_to_search)


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
    
        # --- 3. Process results ---
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
        authors_metadata: List[Any] = []
        if tasks:
            logging.info(f"paper_author_search: Running {len(tasks)} author search tasks concurrently...")
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for idx, result in enumerate(results):
                current_paper_id = ids_to_search[idx]
                if isinstance(result, Exception):
                    logging.error(f"paper_author_search: Task for paper '{current_paper_id}' failed: {result}", exc_info=False)
                    self.not_found_nodes['paper_author'].add(current_paper_id)
                
                elif isinstance(result, PaginatedResults) and hasattr(result, '_items') and result._items: # Result is PaginatedResult
                    authors_metadata.extend(result._items)
                elif isinstance(result, list): # Should ideally be PaginatedResult, but handle list just in case
                    authors_metadata.extend(result)
                
                else:
                    logging.warning(f"paper_author_search: Task for paper '{current_paper_id}' returned no results or failed silently.")
                    self.not_found_nodes['paper_author'].add(current_paper_id) # Mark as not found

        # --- 4. Update global status ---
        self.explored_nodes['paper_author'].update(ids_to_search)
        
        # split authors and papers
        if authors_metadata:
            authors_dict = [author_item.raw_data for author_item in authors_metadata 
                            if isinstance(author_item, Author) and hasattr(author_item, 'raw_data')]
            self.data_pool['author'].extend(authors_dict)


    async def reference_search(
            self,
            paper_ids: List[str],
            citation_limit: Optional[int] = 100,
        ):
        """
        Fetches papers referenced *by* the given paper_ids.
        Adds raw paper metadata for the *referenced* papers to the pool (deduplicated).
        Adds citation relationship data to self.data_pool['reference'].
        """
        citation_limit = citation_limit if citation_limit is not None else self.citation_limit
        # --- 1. Filter ---
        ids_to_search = [pid for pid in paper_ids if pid and pid not in self.explored_nodes['reference']]
        if not ids_to_search:
            logging.info("reference_search: No new paper IDs to search for references.")
            return

        logging.info(f"reference_search: Fetching references for {len(ids_to_search)} papers (limit per paper: {citation_limit}).")

        # --- 2. Conduct search via Semantic Scholar ---
        tasks = []
        for pid in ids_to_search:
            # return a list of references
            tasks.append(self.s2.get_paper_references(paper_id=pid, limit=citation_limit))
                
        # --- 3. Execute and collect ---
        papers_added_count = 0
        rels_added_count = 0
        if tasks:
            logging.info(f"reference_search: Running {len(tasks)} reference search tasks concurrently...")
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for idx, result in enumerate(results):
                source_paper_id = ids_to_search[idx] # The paper whose references we are fetching
                if isinstance(result, Exception):
                    logging.error(f"reference_search: Task for paper '{source_paper_id}' failed: {result}", exc_info=False)
                    self.not_found_nodes['reference'] = source_paper_id

                elif isinstance(result, PaginatedResults) and result._items:
                    logging.info(f"reference_search: Retrieving reference for paper '{source_paper_id}'")
                    for ref_item in result._items:
                        if not isinstance(ref_item, Reference) or not hasattr(ref_item, 'raw_data'): continue

                        # get paper
                        ref_paper = ref_item.paper if hasattr(ref_item, 'paper') else None # The paper being referenced
                        if (isinstance(ref_paper, Paper) and hasattr(ref_paper, 'raw_data') and ref_paper.paperId):
                            self.data_pool['paper'].append(ref_paper.raw_data)
                            papers_added_count += 1

                            # Create citation relationship data
                            ref_dict = {
                                k: v for k, v in ref_item.raw_data.items()
                                if k not in ['citedPaper', 'citingPaper'] and v is not None
                            }
                            ref_dict['source_id'] = source_paper_id
                            ref_dict['target_id'] = ref_paper.paperId
                            self.data_pool['reference'].append(ref_dict)
                            rels_added_count += 1
                        else:
                            logging.debug(f"reference_search: Skipping reference from {source_paper_id} - missing valid citedPaper object: {ref_item.raw_data.get('citedPaper')}")

                else:
                    logging.warning(f"reference_search: Task for paper '{source_paper_id}' returned no results or failed silently.")
                    self.not_found_nodes['reference'].add(source_paper_id)

            logging.info(f"reference_search: Added {papers_added_count} new unique referenced papers and {rels_added_count} reference relationships.")

        # --- 4. Update global status ---
        self.explored_nodes['reference'].update(ids_to_search)


    async def citing_search(
            self,
            paper_ids: Optional[List]=[],
            citation_limit: Optional[int] = 100,
        ):
        """
        Fetches papers that cite the given paper_ids.
        Adds raw paper metadata for the *citing* papers to the pool (deduplicated).
        Adds citation relationship data to self.data_pool['citing'].
        """
        citation_limit = citation_limit if citation_limit is not None else self.citation_limit
        # --- 1. Filter ---
        ids_to_search = [pid for pid in paper_ids if pid and pid not in self.explored_nodes['citing']]
        if not ids_to_search:
            logging.info("citing_search: No new paper IDs to search for citations.")
            return

        logging.info(f"citing_search: Fetching citations for {len(ids_to_search)} papers (limit per paper: {citation_limit}).")

        # --- 2. Create tasks ---
        tasks = []
        logging.info(f"Preparing citing for {len(ids_to_search)} papers ...")
        for pid in ids_to_search:
            # return a list of references
            tasks.append(self.s2.get_paper_citations(paper_id=pid, limit=citation_limit))
                
        # --- 3. Execute and collect ---
        papers_added_count = 0
        rels_added_count = 0
        # Run all query tasks concurrently
        if tasks:
            logging.info(f"citing_search: Running {len(tasks)} citation search tasks concurrently...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for idx, result in enumerate(results):
                target_paper_id = ids_to_search[idx] # The paper being cited
                if isinstance(result, Exception):
                    logging.error(f"citing_search: Task for paper '{target_paper_id}' failed: {result}", exc_info=False)
                    self.not_found_nodes['citing'].add(target_paper_id)

                elif isinstance(result, PaginatedResults) and result._items:
                    for cit_item in result._items:
                        if not isinstance(cit_item, Citation) or not hasattr(cit_item, 'raw_data'): continue
                        
                        # Get the citing paper
                        citing_paper = cit_item.paper if hasattr(cit_item, 'paper') else None
                        if isinstance(citing_paper, Paper) and hasattr(citing_paper, 'raw_data') and citing_paper.paperId:
                            self.data_pool['paper'].append(citing_paper.raw_data)
                            papers_added_count += 1
                        
                            # drop papers in citing
                            cit_dict = {
                                k: v for k, v in cit_item.raw_data.items()
                                if k not in ['citedPaper', 'citingPaper'] and v is not None
                            }
                            cit_dict['source_id'] = citing_paper.paperId 
                            cit_dict['target_id'] = target_paper_id
                            self.data_pool['citing'].append(cit_dict)
                            rels_added_count += 1
                        else:
                            logging.debug(f"citing_search: Skipping citation for {target_paper_id} - missing valid citingPaper object: {cit_item.raw_data.get('citingPaper')}")
                            
                else:
                    logging.warning(f"citing_search: Task for paper '{target_paper_id}' returned no results or failed silently.")
                    self.not_found_nodes['citing'].add(target_paper_id)

            logging.info(f"citing_search: Added {papers_added_count} new unique citing papers and {rels_added_count} citation relationships.")

        # --- 4. Update global status ---
        # update explored nodes
        self.explored_nodes['citing'].update(ids_to_search)


    async def paper_recommendation(
            self,
            pos_paper_ids: List[str],
            neg_paper_ids: Optional[List] = [],
            recommend_limit: Optional[int] = 100,
        ):
        """
        Fetches paper recommendations based on positive and negative paper IDs.
        Adds recommended raw paper metadata to the pool (deduplicated).
        """
        neg_paper_ids = neg_paper_ids or []
        recommend_limit = recommend_limit if recommend_limit is not None else self.recommend_limit
        # Create a unique, hashable key for explored check
        recommendation_key = (tuple(sorted(pos_paper_ids)), tuple(sorted(neg_paper_ids)))

        # --- 1. Filter ---
        if not pos_paper_ids:
            logging.error("paper_recommendation: No positive paper IDs provided.")
            return

        logging.info(f"paper_recommendation: Fetching {recommend_limit} recommendations based on {len(pos_paper_ids)} positive and {len(neg_paper_ids)} negative papers.")
        
        # --- 2. Get recommend papers ---
        # search returns a list of papers
        papers_metadata = await self.s2.get_recommended_papers_from_lists(
            positive_paper_ids=pos_paper_ids,
            negative_paper_ids=neg_paper_ids,
            limit=recommend_limit)

        # --- 3. Update global status ---
        # update explored nodes
        self.explored_nodes['recommendation'].add(recommendation_key)

        # add paper metadata to data pool
        if papers_metadata:
            papers_dict = [item.raw_data for item in papers_metadata if isinstance(item, Paper)]
            self.data_pool['paper'].extend(papers_dict)
