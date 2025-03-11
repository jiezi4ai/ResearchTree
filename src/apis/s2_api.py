# Code originated from Daniel Silva's project [semanticscholar](https://github.com/danielnsilva/semanticscholar)
# This only serves as a wrapper of some key functions.

import math
import time
import random
from semanticscholar import SemanticScholar  # pip install semanticscholar 
from typing import List, Dict

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
            batch_cnt = math.ceil(id_cnt / 500)
            for i in range(batch_cnt):
                batch_ids = id_list[i*500:(i+1)*500]
                batch_results = self.scholar.get_papers(paper_ids=batch_ids, fields=fields)
                for item in batch_results:
                    paper_metadata.append(item.__dict__.get('_data', {}))
                time.sleep(random.uniform(10, 15))
        return paper_metadata


    def search_author_by_ids(self, author_ids, fields: list = None):
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
            batch_cnt = math.ceil(id_cnt / 500)
            for i in range(batch_cnt):
                batch_ids = id_list[i*500:(i+1)*500]
                batch_results = self.scholar.get_authors(author_ids=batch_ids, fields=fields)
                for item in batch_results:
                    # convert Author object to dict
                    item = item.__dict__.get('_data', {})
                    author_metadata.append(item)
                time.sleep(random.uniform(10, 15))
        return author_metadata


    # get paper by search
    def search_paper_by_keywords(
        self,
        query: str,
        year: str = None,
        publication_types: list = None,
        open_access_pdf: bool = None,
        venue: list = None,
        fields_of_study: list = None,
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
                fields_of_study=fields_of_study,
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
        limit: int = 100
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
        results = self.scholar.get_paper_references(paper_id, fields,limit)

        refs_metadata = []
        for item in results[0:limit]:
            refs_metadata.append(item.__dict__.get('_data', {}))
        return refs_metadata

        
    # get papers citing this paper
    def get_s2_citing_papers(
        self,  
        paper_id: str,
        fields: list = None,
        limit: int = 100
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
        results = self.scholar.get_paper_citations(paper_id, fields, limit)
    
        citedby_metadata = []
        for item in results[0:limit]:
            citedby_metadata.append(item.__dict__.get('_data', {}))
        return citedby_metadata


    def get_s2_recommended_papers(
        self,
        positive_paper_ids: List[str],
        negative_paper_ids: List[str] = None,
        fields: list = None,
        limit: int = 100
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
        results = self.scholar.get_recommended_papers_from_lists(
            positive_paper_ids, negative_paper_ids, fields, limit
            )
        
        rec_metadata = []
        for item in results[0:limit]:
            rec_metadata.append(item.__dict__.get('_data', {}))
        return rec_metadata