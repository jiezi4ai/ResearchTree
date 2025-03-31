# Code originated from Daniel Silva's project [semanticscholar](https://github.com/danielnsilva/semanticscholar)
# key updates:
# 1. align output to dict format
# 2. add error handler
# 3. retrieve abstract for papers from reference, from recommendation or from author information 

import math
import time
import random
from semanticscholar import SemanticScholar  # pip install semanticscholar 
from typing import List, Dict, Optional

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SemanticScholarKit:
    def __init__(self, ss_api_key: str = None, ss_api_url: str = None):
        """
        :param str ss_api_key: (optional) private API key.
        :param str ss_api_url: (optional) custom API url.
        """
        self.scholar = SemanticScholar(api_key=ss_api_key, api_url=ss_api_url)


    # get paper information
    def search_paper_by_ids(self, id_list, fields: list = None) -> List[Dict]:
        """search paper by id
        Args:
            :param str paper_id: S2PaperId, CorpusId, DOI, ArXivId, MAG, ACL, 
                PMID, PMCID, or URL from:
                - semanticscholar.org
                - arxiv.org
                - aclweb.org
                - acm.org
                - biorxiv.org
            :param list fields: (optional) list of the fields to be returned.
        Returns:
            :returns: paper metadata in list of dicts
                dict_keys(['paperId', 'externalIds', 'corpusId', 'publicationVenue', 'url', 'title', 'abstract', 'venue', 'year', 
                           'referenceCount', 'citationCount', 'influentialCitationCount', 'isOpenAccess', 'openAccessPdf', 
                           'fieldsOfStudy', 's2FieldsOfStudy', 'publicationTypes', 'publicationDate', 'journal', 'citationStyles', 
                           'authors'])
        Note:
            Refer to get_papers function.
        """
        id_list = [x for x in id_list if x and type(x)==str]
        id_cnt = len(id_list)

        paper_metadata = []
        if id_cnt > 0:
            batch_cnt = math.ceil(id_cnt / 100)
            for i in range(batch_cnt):
                batch_ids = id_list[i*100:(i+1)*100]
                try:
                    batch_results = self.scholar.get_papers(paper_ids=batch_ids, fields=fields)
                except Exception as e:
                    logger.error(f"Error when getting papers for batch {i}: {e}")
                    batch_results = []
                for item in batch_results:
                    paper_metadata.append(item.__dict__.get('_data', {}))
                time.sleep(random.uniform(10, 15))
        return paper_metadata


    def search_author_by_ids(
            self, 
            author_ids, 
            fields: list = None,
            with_abstract: Optional[bool]=False
            ):
        """search author by ids
        Args:
            :param str author_ids: list of S2AuthorId (must be <= 1000).
        Returns:
            :returns: author data, and optionally list of IDs not found.
            :rtype: :class:`List` of :class:`semanticscholar.Author.Author` 
                    or :class:`Tuple` [:class:`List` of 
                    :class:`semanticscholar.Author.Author`, 
                    :class:`List` of :class:`str`]
            :raises: BadQueryParametersException: if no author was found.
        """
        id_list = [x for x in author_ids if x and type(x)==str]
        id_cnt = len(id_list)

        author_metadata = []
        if id_cnt > 0:
            batch_cnt = math.ceil(id_cnt / 100)
            for i in range(batch_cnt):
                batch_ids = id_list[i*100:(i+1)*100]
                try:
                    batch_results = self.scholar.get_authors(author_ids=batch_ids, fields=fields)
                except Exception as e:
                    logger.error(f"Error when getting authors for batch {i}: {e}")
                    batch_results = []
                for item in batch_results:
                    # convert Author object to dict
                    item = item.__dict__.get('_data', {})
                    author_metadata.append(item)
                time.sleep(random.uniform(10, 15))

        if with_abstract:
            # filter paper ids with missing abstract
            tmp_ids = []
            for info in author_metadata:
                papers = info.get('papers', [])
                for paper in papers:
                    paper_id = paper.get('paperId')
                    abstract = paper.get('abstract')
                    if paper_id is not None and abstract is None:
                        tmp_ids.append(paper_id)
            
            # search again for paper with missing abstract
            if len(tmp_ids) > 0:
                paper_metadata = self.search_paper_by_ids(id_list=tmp_ids, fields=fields)
            
            # retrieve abstract and update original information
            ref_abstracts = {item['paperId']: item['abstract'] for item in paper_metadata 
                             if item.get('paperId') is not None and item.get('abstract') is not None}
            
            for info in author_metadata:
                papers = info.get('papers', [])
                for paper in papers:
                    paper_id = paper.get('paperId')
                    abstract = paper.get('abstract')
                    if paper_id is not None and abstract is None:
                        paper['abstract'] = ref_abstracts.get(paper_id)

        return author_metadata


    # get paper by search
    def search_paper_by_keywords(
        self,
        query: str,
        year: str = None,
        publication_types: list = None,
        open_access_pdf: bool = None,
        venue: list = None,
        fields: list = None,
        publication_date_or_year: str = None,
        min_citation_count: int = None,
        limit: int = 100,
        bulk: bool = False,
        sort: str = None,
        match_title: bool = False
    ) -> List[Dict]:
        """
        Search for papers by keyword. Performs a search query based on the 
        S2 search relevance algorithm, or a bulk retrieval of basic paper 
        data without search relevance (if bulk=True). Paper relevance 
        search is the default behavior and returns up to 1,000 results. 
        Bulk retrieval instead returns up to 10,000,000 results (1,000 
        in each page).
        Args:
            :param str query: plain-text search query string.
            :param str year: (optional) restrict results to the given range of publication year.
                Examples: '2000', '1991-2000', '1991-', '-2000'.
            :param list publication_type: (optional) restrict results to the given publication type list.
                Examples: 'Review', 'JournalArticle'. Reference from https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/get_graph_paper_relevance_search
            :param bool open_access_pdf: (optional) restrict results to papers with public PDFs.
            :param list venue: (optional) restrict results to the given venue list.
            :param list fields_of_study: (optional) restrict results to given field-of-study list, using the s2FieldsOfStudy paper field.
                Examples: 'Computer Science', 'Physics,Mathematics'. Reference from https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/get_graph_paper_relevance_search.
            :param str publication_date_or_year: (optional) restrict results to 
                the given range of publication date in the format 
                <start_date>:<end_date>, where dates are in the format 
                YYYY-MM-DD, YYYY-MM, or YYYY.
            :param int min_citation_count: (optional) restrict results to papers with at least the given number of citations.
            :param int limit: (optional) maximum number of results to return (must be <= 100).
            :param bool bulk: (optional) bulk retrieval of basic paper data 
                without search relevance (ignores the limit parameter if True 
                and returns up to 1,000 results in each page).
            :param str sort: (optional) sorts results (only if bulk=True) using 
                <field>:<order> format, where "field" is either paperId, 
                publicationDate, or citationCount, and "order" is asc 
                (ascending) or desc (descending).
            :param bool match_title: (optional) retrieve a single paper whose title best matches the given query.
        Returns:
            :returns: paper metadata in list dicts.
                dict_keys(['paperId', 'externalIds', 'corpusId', 'publicationVenue', 'url', 'title', 'abstract', 'venue', 'year', 
                           'referenceCount', 'citationCount', 'influentialCitationCount', 'isOpenAccess', 'openAccessPdf', 
                           'fieldsOfStudy', 's2FieldsOfStudy', 'publicationTypes', 'publicationDate', 'journal', 'citationStyles', 
                           'authors'])
        Note:
            Refer to search_paper function.
        """
        max_result = min(limit, 100)
        results = self.scholar.search_paper(query=query,
                year=year,
                publication_types=publication_types,
                open_access_pdf=open_access_pdf,
                venue=venue,
                fields_of_study=fields,
                publication_date_or_year=publication_date_or_year,
                min_citation_count=min_citation_count,
                limit=max_result,
                bulk=bulk,
                sort=sort,
                match_title=match_title)
        
        paper_metadata = []
        if results.total > 0:
            for item in results[0:max_result]:
                paper_metadata.append(item.__dict__.get('_data', {}))  
        return paper_metadata


    # get all papers cited by this paper
    def get_s2_cited_papers(
        self,
        paper_id: str,
        fields: list = None,
        limit: int = 100,
        with_abstract: Optional[bool]=False
    ) -> List[Dict]:
        """Get papers cited by this paper
        Args:
            :param str paper_id: S2PaperId, CorpusId, DOI, ArXivId, MAG, ACL, PMID, PMCID, or URL from:
                - semanticscholar.org
                - arxiv.org
                - aclweb.org
                - acm.org
                - biorxiv.org
            :param list fields: (optional) list of the fields to be returned.
            :param int limit: (optional) maximum number of results to return (must be <= 1000).
        Returns:
            :returns: reference metadata in list dicts.
                dict_keys(['contextsWithIntent',  # orginal context within the paper
                           'isInfluential',  # indicate if the cited paper important
                           'contexts', 
                           'intents',   # show which section the citation is
                           'citedPaper'  # cited paper metadata
                           ])
        Note:
            Null data for newly released papers.
        """
        try:
            results = self.scholar.get_paper_references(paper_id, fields, limit)
        except Exception as e:
            logger.error(f"Error when getting papers cited by {paper_id}: {e}")
            results = []

        refs_metadata = []
        for item in results[0:limit]:
            refs_metadata.append(item.__dict__.get('_data', {}))

        if with_abstract:
            # filter paper ids with missing abstract
            tmp_ids = []
            for info in refs_metadata:
                paper = info.get('citedPaper')
                if isinstance(paper, dict):
                    paper_id = paper.get('paperId')
                    abstract = paper.get('abstract')
                    if paper_id is not None and abstract is None:
                        tmp_ids.append(paper_id)
            
            # search again for paper with missing abstract
            if len(tmp_ids) > 0:
                paper_metadata = self.search_paper_by_ids(id_list=tmp_ids, fields=fields)
            
            # retrieve abstract and update original information
            ref_abstracts = {item['paperId']: item['abstract'] for item in paper_metadata 
                             if item.get('paperId') is not None and item.get('abstract') is not None}
            
            for info in refs_metadata:
                paper = info.get('citedPaper')
                if paper is not None:
                    paper_id = paper.get('paperId')
                    abstract = paper.get('abstract')
                    if paper_id is not None and abstract is None:
                        paper['abstract'] = ref_abstracts.get(paper_id)

        return refs_metadata

        
    # get papers citing this paper
    def get_s2_citing_papers(
        self,  
        paper_id: str,
        fields: list = None,
        limit: int = 100,
        with_abstract: Optional[bool]=False
    ) -> List[Dict]:
        """
        Get all papers citing this paper
        Args:
            :param str paper_id: S2PaperId, CorpusId, DOI, ArXivId, MAG, ACL, PMID, PMCID, or URL from:
                - semanticscholar.org
                - arxiv.org
                - aclweb.org
                - acm.org
                - biorxiv.org
            :param list fields: (optional) list of the fields to be returned.
            :param int limit: (optional) maximum number of results to return (must be <= 1000).
        Returns:
            :returns: reference metadata in list dicts.
                dict_keys(['contextsWithIntent',  # orginal context within the paper
                           'isInfluential',  # indicate if the cited paper important
                           'contexts', 
                           'intents',   # show which section the citation is
                           'citingPaper'  # cited paper metadata
                           ])
        """
        try:
            results = self.scholar.get_paper_citations(paper_id, fields, limit)
        except Exception as e:
            logger.error(f"Error when getting papers citing {paper_id}: {e}")
            results = []
    
        citedby_metadata = []
        for item in results[0:limit]:
            citedby_metadata.append(item.__dict__.get('_data', {}))

        if with_abstract:
            # filter paper ids with missing abstract
            tmp_ids = []
            for info in citedby_metadata:
                paper = info.get('citingPaper')
                if isinstance(paper, dict):
                    paper_id = paper.get('paperId')
                    abstract = paper.get('abstract')
                    if paper_id is not None and abstract is None:
                        tmp_ids.append(paper_id)
            
            # search again for paper with missing abstract
            if len(tmp_ids) > 0:
                paper_metadata = self.search_paper_by_ids(id_list=tmp_ids, fields=fields)
            
            # retrieve abstract and update original information
            ref_abstracts = {item['paperId']: item['abstract'] for item in paper_metadata 
                             if item.get('paperId') is not None and item.get('abstract') is not None}
            
            for info in citedby_metadata:
                paper = info.get('citingPaper')
                if paper is not None:
                    paper_id = paper.get('paperId')
                    abstract = paper.get('abstract')
                    if paper_id is not None and abstract is None:
                        paper['abstract'] = ref_abstracts.get(paper_id)

        return citedby_metadata


    def get_s2_recommended_papers(
        self,
        positive_paper_ids: List[str],
        negative_paper_ids: List[str] = None,
        fields: list = None,
        limit: int = 100,
        with_abstract: Optional[bool]=False
    ) -> List[Dict]:
        """Get recommended papers for lists of positive and negative examples.
        Args:
            :param list positive_paper_ids: list of paper IDs 
                that the returned papers should be related to.
            :param list negative_paper_ids: (optional) list of paper IDs 
                that the returned papers should not be related to.
            :param list fields: (optional) list of the fields to be returned.
            :param int limit: (optional) maximum number of recommendations to 
                return (must be <= 500).
        Returns:
            :returns: list of recommendations.
        """
        try:
            results = self.scholar.get_recommended_papers_from_lists(
                positive_paper_ids, negative_paper_ids, fields, limit
                )
        except Exception as e:
            logger.error(f"Error when getting recommended papers regarding {positive_paper_ids}: {e}")
            results = []
        
        rec_metadata = []
        for item in results[0:limit]:
            rec_metadata.append(item.__dict__.get('_data', {}))

        if with_abstract:
            # filter paper ids with missing abstract
            tmp_ids = []
            for paper in rec_metadata:
                paper_id = paper.get('paperId')
                abstract = paper.get('abstract')
                if paper_id is not None and abstract is None:
                    tmp_ids.append(paper_id)
            
            # search again for paper with missing abstract
            if len(tmp_ids) > 0:
                paper_metadata = self.search_paper_by_ids(id_list=tmp_ids, fields=fields)
            
            # retrieve abstract and update original information
            ref_abstracts = {item['paperId']: item['abstract'] for item in paper_metadata 
                             if item.get('paperId') is not None and item.get('abstract') is not None}
            
            for paper in rec_metadata:
                paper_id = paper.get('paperId')
                abstract = paper.get('abstract')
                if paper_id is not None and abstract is None:
                    paper['abstract'] = ref_abstracts.get(paper_id)

        return rec_metadata