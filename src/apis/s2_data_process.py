import re
from typing import Optional, List, Dict, Literal
from semanticscholar import Paper, Author

from utils.data_process import generate_hash_key, remove_kth_element, remove_key_values, filter_and_reorder_dict

def align_paper_metadata(s2_papers: List[Dict]|Dict):
    """convert reset id for paper metadata (from semantic scholar)"""
    if isinstance(s2_papers, dict):
        s2_papers = [s2_papers]

    s2_papers_dict = []
    for item in s2_papers:
        # set up paper id
        s2_paper_id = item.get('paperId')
        
        if s2_paper_id is not None:
            doi = item.get('externalIds',{}).get('DOI')  # doi
            arxiv_id = item.get('externalIds',{}).get('ArXiv')  # arxiv id

            if doi is not None and doi.startswith('10.48550/arXiv.') and arxiv_id is None:
                arxiv_id = doi

            arxiv_id_rvsd = None
            if arxiv_id is not None:
                arxiv_no = arxiv_id.replace('10.48550/arXiv.', '')   # get rid of doi prefix
                arxiv_id_rvsd = re.sub(r'v\d+$', '', arxiv_no)   # get rid of version info
                version_match = re.search(r'v\d+$', arxiv_no)  # get version info
                # generate arxiv related info
                item['version'] = version_match.group(0) if version_match else ""
                item['arxivUrl'] = f"https://arxiv.org/abs/{arxiv_no}"
                item['isOpenAccess'] = True
                item['openAccessPdf'] = f"https://arxiv.org/pdf/{arxiv_no}"
                item['arxivId'] = arxiv_id_rvsd

            if doi is None:
                if arxiv_id_rvsd is not None:
                    doi = f"10.48550/arXiv.{arxiv_id_rvsd}"  # assign 10.48550/arXiv. for arxiv id https://info.arxiv.org/help/doi.html
                else:
                    doi = s2_paper_id
            item['doi'] = doi

            # process publish date
            publish_dt = item.get('publicationDate')
            year = item.get('year')
            if publish_dt is None:
                if arxiv_id_rvsd is not None:
                    item['publicationDate'] = f"20{arxiv_id_rvsd[:2]}-{arxiv_id_rvsd[2:4]}-01"
                elif year is not None:
                    item['publicationDate'] = f"{year}-01-01"
                else:
                    item['publicationDate'] = '2000-01-01'

            # cut down author information
            item['authors'] = item.get('authors', [])[0:10]

            s2_papers_dict.append(item)
    return s2_papers_dict
            

def review_and_filter_paper(
    s2_papers: List[Paper.Paper]|Paper.Paper,
    from_dt: Optional[str] = None,   # filter publish dt no earlier than
    to_dt: Optional[str] = None,   # filter publish dt no late than
    fields_of_study: Optional[List[str]] = None,  # list of field of study
    min_citation_cnt: Optional[int] = 0,  # citation count no less than
    author_ids: Optional[List[str]] = None,  # restrcted to list of authors' ids
    # institutions: Optional[List[str]] = None,  # restrcted to list of institutions, to be implemented
    # journals: Optional[List[str]] = None,  # restrcted to list of journals, to be implemented
):
    """filter paper metadata (from semantic scholar) based on given criteria"""
    papers_dict = align_paper_metadata(s2_papers)
    review_result = {
        'papers_filtered': {},  # k-v dict
        'off_dt_range': {},
        'off_fields_of_study': {},
        'below_min_citation': {},
        'off_author_scope': {},
        'lack_abstract': {},
    }

    filter_fos_set = set(fields_of_study) if fields_of_study is not None else None
    filter_author_id_set = set(author_ids) if author_ids is not None else None
    for item in papers_dict:
        paper_doi = item.get('doi')
        if paper_doi is None:
            continue

        publish_dt = item.get('publicationDate')
        paper_author_ids_set = set(x.get('authorId') for x in item.get('authors', []) if x.get('authorId') is not None)
        paper_fos_set = set(item.get('fieldsOfStudy', []))
        paper_citation_cnt = item.get('referenceCount', 0)
        paper_abstract = item.get('abstract')
        
        # --- Filtering Logic ---
        reject = False
        # exclude paper out of time scope
        if from_dt is not None and to_dt  is not None and (publish_dt < from_dt or publish_dt > to_dt):  
            reject = True
            review_result['off_dt_range'][paper_doi] = item
        
        # exclude paper not in fields of study
        if filter_fos_set is not None and paper_fos_set is not None and not filter_fos_set.intersection(paper_fos_set): 
            reject = True
            review_result['off_fields_of_study'][paper_doi] = item

        # exclude paper not meeting citation criteria
        if min_citation_cnt is not None and paper_citation_cnt < min_citation_cnt:   
            reject = True
            review_result['below_min_citation'][paper_doi] = item
        
        # exclude paper not in author list
        if filter_author_id_set is not None and not filter_author_id_set.intersection(paper_author_ids_set):
            reject = True
            review_result['off_author_scope'][paper_doi] = item

        if not reject:
            review_result['papers_filtered'][paper_doi] = item

        # --- Independent Abstract Check ---
        # identify paper without abstract
        if paper_abstract is None or len(paper_abstract) < 10:
            review_result['lack_abstract'][paper_doi] = item

    return review_result


def process_paper_data(papers_filtered: List[Dict]|Dict):
    """standardize paper metadata to better suit neo4j format
    Argss:
        papers_filtered ([List[Dict]|Dict]): paper metadata that has been reset and filtered
    Returns:
        - the json has to be preprocessed to in format like:
                        [{'type': 'node',
                        'id': '2345003971',
                        'labels': ['Author'],
                        'properties': {'authorId': '2345003971', 'name': 'Mark Schone'}},
                        {'type': 'relationship',
                        'relationshipType': 'WRITES',
                        'startNodeId': '2345003971',
                        'endNodeId': '10.48550/arXiv.2502.07827',
                        'properties': {'authorOrder': 1,
                        'coauthors': [{'authorId': '2345003971', 'name': 'Mark Schone'},]}}]    
    """
    if isinstance(papers_filtered, dict):
        papers_filtered = [papers_filtered]

    # then process to standard json format
    s2_paper_json = []
    processed_node_ids = set() # set of node ids
    processed_edge_tuples = set() # set of tuples for edge ids
    
    for item in papers_filtered:
        paper_doi = item.get('doi') 
        if paper_doi is None:
            continue 

        # --- Process Paper Node ---
        paper_props = ['doi', 'title', 'abstract', 'year', 'publicationDate',
                        'citationCount', 'referenceCount', 'influentialCitationCount', 'arxivId', 'arxivUrl',
                        'isOpenAccess', 'openAccessPdf', 'version', 'paperId',
                        'fieldsOfStudy', 'authors'] # Add/remove as needed
        paper_properties = {k: v for k, v in item.items() if k in paper_props and v is not None}
        paper_node = {
            "type": "node",
            "id": paper_doi,
            "labels": ["Paper"],
            "properties": paper_properties # Use filtered properties
            }
        s2_paper_json.append(paper_node)
        processed_node_ids.add(paper_doi)

        # --- Process Authors ---
        authors = item.get('authors', [])[:10]
        for idx, author in enumerate(authors):
            # process author node
            author_id = author.get('authorId')
            if author_id is not None:
                if author_id not in processed_node_ids:
                    author_node = {
                        "type": "node",
                        "id": author.get('authorId'),
                        "labels": ["Author"],
                        "properties": author}
                    s2_paper_json.append(author_node)
                    processed_node_ids.add(author_id)
            
                # process author -> WRITES -> paper
                edge_tuple = (author_id, paper_doi, "WRITES")
                if edge_tuple not in processed_edge_tuples:
                    author_order = idx + 1
                    coauthors = remove_kth_element(authors, idx)   
                    author_paper_relationship = {
                        "type": "relationship",
                        "relationshipType": "WRITES",
                        "startNodeId": author_id,
                        "endNodeId": paper_doi,
                        "properties": {'authorOrder': author_order, 'coauthors': coauthors}
                        }
                    s2_paper_json.append(author_paper_relationship)
                    processed_edge_tuples.add(edge_tuple)

        # --- Process Journal ---
        journal = item.get('journal', {})
        journal_name = journal.get('name') if isinstance(journal, dict) else None
        if journal_name is not None:
            journal_hash_id = generate_hash_key(journal_name)
            if journal_hash_id not in processed_node_ids:
                journal_node = {
                    "type": "node",
                    "id": journal_hash_id,
                    "labels": ["Journal"],
                    "properties": {"journal_hash_id": journal_hash_id, "name": journal_name, "hash_method":"hashlib.sha256"}}
                s2_paper_json.append(journal_node)
                processed_node_ids.add(journal_hash_id)
            
            # process paper -> PRINTS_ON -> journal
            edge_tuple = (paper_doi, journal_hash_id, "PRINTS_ON")
            if edge_tuple not in processed_edge_tuples:
                if 'arxiv' not in journal_name.lower():  # exclude arxiv from journal
                    paper_journal_relationship = {
                        "type": "relationship",
                        "relationshipType": "PRINTS_ON",
                        "startNodeId": paper_doi,
                        "endNodeId": journal_hash_id,
                        "properties": {}}
                    s2_paper_json.append(paper_journal_relationship)
                    processed_edge_tuples.add(edge_tuple)

        # --- Process Venue ---
        venue = item.get('publicationVenue', {})
        venue_id = venue.get('id') if isinstance(venue, dict) else None
        if venue_id is not None:
            if venue_id not in processed_node_ids:
                venue_node = {
                    "type": "node",
                    "id": venue_id,
                    "labels": ["Venue"],
                    "properties": venue
                    }
                s2_paper_json.append(venue_node)
                processed_node_ids.add(venue_id)
            
            # process paper -> RELEASES_IN -> venue
            edge_tuple = (paper_doi, venue_id, "RELEASES_IN")
            if edge_tuple not in processed_edge_tuples:
                if 'arxiv' not in venue.get('name', '').lower():  # exclude arxiv from venue
                    paper_venue_relationship = {
                        "type": "relationship",
                        "relationshipType": "RELEASES_IN",
                        "startNodeId": paper_doi,
                        "endNodeId": venue_id,
                        "properties": {}}
                    s2_paper_json.append(paper_venue_relationship)
                    processed_edge_tuples.add(edge_tuple)

    return s2_paper_json


def process_author_data(
        s2_authors: List[Dict]|Dict,
        from_dt: Optional[str] = None,   # filter publish dt no earlier than
        to_dt: Optional[str] = None,   # filter publish dt no late than
        fields_of_study: Optional[List[str]] = None,  # list of field of study
        # min_citation_cnt: Optional[int] = 0,  # citation count no less than
        # institutions: Optional[List[str]] = None,  # restrcted to list of institutions, to be implemented
        # journals: Optional[List[str]] = None,  # restrcted to list of journals, to be implemented
        # author_ids: Optional[List[str]] = None,  # restrcted to list of authors' ids      
    ):
    """standardize author metadata to better suit neo4j format
    Argss:
        s2_authors ([List[Dict]|Dict]): author metadata
    Returns:
        - the json has to be preprocessed to in format like:
                        [{'type': 'node',
                        'id': '2345003971',
                        'labels': ['Author'],
                        'properties': {'authorId': '2345003971', 'name': 'Mark Schone'}},
                        {'type': 'relationship',
                        'relationshipType': 'WRITES',
                        'startNodeId': '2345003971',
                        'endNodeId': '10.48550/arXiv.2502.07827',
                        'properties': {'authorOrder': 1,
                        'coauthors': [{'authorId': '2345003971', 'name': 'Mark Schone'},]}}]    
    """
    if isinstance(s2_authors, dict):
        s2_authors = [s2_authors]

    s2_author_json = []
    processed_node_ids = set() # set of node ids
    processed_edge_tuples = set() # set of tuples for edge ids

    for item in s2_authors:
        author_id = item.get('authorId')
        if author_id is not None:
            continue
        
        author_id = item.get('authorId')
        if author_id is not None:
            # process author metadata
            author_node = {
                        "type": "node",
                        "id": author_id,
                        "labels": ["Author"],
                        "properties": remove_key_values(item, 'papers')}
            s2_author_json.append(author_node)
            processed_node_ids.add(author_id) 

            # --- Process Affiliations ---
            # process institution metadata
            affiliations = item.get('affiliations', [])
            if isinstance(affiliations, list) and len(affiliations) > 0:
                for inst in affiliations:
                    inst_hash_id = generate_hash_key(inst)
                    if inst_hash_id not in processed_node_ids:
                        inst_node = {
                                    "type": "node",
                                    "id": inst_hash_id,
                                    "labels": ["Affiliation"],
                                    "properties":  {'affiliation_hash_id': inst_hash_id, 'affiliation_name': inst, 'hash_method': 'hashlib.sha256'}}
                        s2_author_json.append(inst_node)  
                        processed_node_ids.add(inst_hash_id)   

                    # process author -> WORKS_IN -> affiliations
                    edge_tuple = (author_id, inst_hash_id, "WORKS_IN")
                    if edge_tuple not in processed_edge_tuples:
                        author_inst_relationship = {
                            "type": "relationship",
                            "relationshipType": "WORKS_IN",
                            "startNodeId": author_id,
                            "endNodeId": inst_hash_id,
                            "properties": {}}
                        s2_author_json.append(author_inst_relationship)
                        processed_edge_tuples.add(edge_tuple)

            # --- Process Paper ---
            # process all paper metadata
            s2_papers = item.get('papers', [])
            if isinstance(s2_papers, list) is not None and len(s2_papers) > 0:
                review_result = review_and_filter_paper(
                            s2_papers=s2_papers,
                            from_dt=from_dt, 
                            to_dt=to_dt, 
                            fields_of_study=fields_of_study)
                papers_filtered = list(review_result.get('papers_filtered').values())
                
                s2_paper_json = process_paper_data(papers_filtered=papers_filtered)

                for x in s2_paper_json:
                    if x['type'] == "node" and x['id'] not in processed_node_ids:
                        s2_author_json.append(x)
                        processed_node_ids.add(x['id'])
                    elif x['type'] == "relationship":
                        edge_tuple = (x['startNodeId'], x['endNodeId'], x['relationshipType'])
                        if edge_tuple not in processed_edge_tuples:
                            s2_author_json.append(x)
                            processed_edge_tuples.add(edge_tuple)

    return s2_author_json


def process_citation_metadata(
        original_paper_doi: str,
        s2_citation_metadata: List[Dict]|Dict,
        citation_type: Literal['citingPaper','citedPaper'],
        from_dt: Optional[str] = None,   # filter publish dt no earlier than
        to_dt: Optional[str] = None,   # filter publish dt no late than
        fields_of_study: Optional[List[str]] = None,  # list of field of study
        min_citation_cnt: Optional[int] = None,  # citation count no less than
        institutions: Optional[List[str]] = None,  # restrcted to list of institutions, to be implemented
        journals: Optional[List[str]] = None,  # restrcted to list of journals, to be implemented
        author_ids: Optional[List[str]] = None,  # restrcted to list of authors' ids      
        ):
    """standardize paper citation relationships to better suit neo4j format
    Argss:
        s2_citation_metadata ([List[Dict]|Dict]): citing or cited by metadata
    Returns:
        - the json has to be preprocessed to in format like:
                        [{'type': 'node',
                        'id': '2345003971',
                        'labels': ['Author'],
                        'properties': {'authorId': '2345003971', 'name': 'Mark Schone'}},
                        {'type': 'relationship',
                        'relationshipType': 'WRITES',
                        'startNodeId': '2345003971',
                        'endNodeId': '10.48550/arXiv.2502.07827',
                        'properties': {'authorOrder': 1,
                        'coauthors': [{'authorId': '2345003971', 'name': 'Mark Schone'},]}}]    
    """
    s2_citationmeta_json = []

    # for citations (citing or cited papers)
    if isinstance(s2_citation_metadata, dict):
        s2_citation_metadata = [s2_citation_metadata] 
    
    for item in s2_citation_metadata:
        existing_node_ids = [x['id'] for x in s2_citationmeta_json if x['type']=='node']
        existing_edge_ids = [(x['startNodeId'],x['endNodeId'])  for x in s2_citationmeta_json if x['type']=='relationship']

        s2_paper_metadata = item.get(citation_type)  # get either citing or cited paper metadata
        
        # convert paper metadata to json
        filtered_s2_paper_metadata = reset_and_filter_paper(
            s2_paper_metadata=s2_paper_metadata,
            from_dt=from_dt, 
            to_dt=to_dt, 
            fields_of_study=fields_of_study)
        s2_papermeta_json = process_paper_metadata(filtered_s2_paper_metadata)

        for x in s2_papermeta_json:
            if x['type'] == "node" and x['id'] not in existing_node_ids:
                s2_citationmeta_json.append(x)
            elif x['type'] == "relationship" and (x['startNodeId'], x['endNodeId']) not in existing_edge_ids:
                s2_citationmeta_json.append(x)

        target_paper_doi = filtered_s2_paper_metadata[0].get('doi') if isinstance(filtered_s2_paper_metadata, list) and len(filtered_s2_paper_metadata) > 0 else None
        if target_paper_doi is not None:
            if citation_type == 'citedPaper':  # source paper citing target papers
                start_node_id = original_paper_doi
                end_node_id = target_paper_doi
            else:  # source paper cited by target papers
                start_node_id = target_paper_doi
                end_node_id = original_paper_doi 

            # append relationship
            if (start_node_id, end_node_id) not in existing_edge_ids:
                properties = filter_and_reorder_dict(item, ['isInfluential', 'contexts', 'intents', 'contextsWithIntent'])
                paper_cites_relationship = {
                    "type": "relationship",
                    "relationshipType": "CITES",
                    "startNodeId": start_node_id,
                    "endNodeId": end_node_id,
                    "properties": properties}
                s2_citationmeta_json.append(paper_cites_relationship)

    return s2_citationmeta_json
    

def process_related_metadata(
        s2_related_metadata: List[Dict]|Dict,
        topic: Optional[str] = None,
        from_dt: Optional[str] = None,   # filter publish dt no earlier than
        to_dt: Optional[str] = None,   # filter publish dt no late than
        fields_of_study: Optional[List[str]] = None,  # list of field of study
        min_citation_cnt: Optional[int] = 0,  # citation count no less than
        institutions: Optional[List[str]] = None,  # restrcted to list of institutions, to be implemented
        journals: Optional[List[str]] = None,  # restrcted to list of journals, to be implemented
        author_ids: Optional[List[str]] = None,  # restrcted to list of authors' ids      
    ):
    """standardize paper citation relationships to better suit neo4j format
    Argss:
        s2_related_metadata ([List[Dict]|Dict]): related papers metadata
        properties (dict): more information on "RELATES" information
    Returns:
        - the json has to be preprocessed to in format like:
                        [{'type': 'node',
                        'id': '2345003971',
                        'labels': ['Author'],
                        'properties': {'authorId': '2345003971', 'name': 'Mark Schone'}},
                        {'type': 'relationship',
                        'relationshipType': 'WRITES',
                        'startNodeId': '2345003971',
                        'endNodeId': '10.48550/arXiv.2502.07827',
                        'properties': {'authorOrder': 1,
                        'coauthors': [{'authorId': '2345003971', 'name': 'Mark Schone'},]}}]    
    """
    s2_relatedmeta_json = []

    # for related papers
    if isinstance(s2_related_metadata, dict):
        s2_related_metadata = [s2_related_metadata]
    
    for paper_metadata in s2_related_metadata:
        existing_node_ids = [x['id'] for x in s2_relatedmeta_json if x['type']=='node']
        existing_edge_ids = [(x['startNodeId'],x['endNodeId'])  for x in s2_relatedmeta_json if x['type']=='relationship']

        # convert paper metadata to json
        filtered_s2_paper_metadata = reset_and_filter_paper(
            s2_paper_metadata=paper_metadata,
            from_dt=from_dt, 
            to_dt=to_dt, 
            fields_of_study=fields_of_study)
        s2_papermeta_json = process_paper_metadata(filtered_s2_paper_metadata)
        for x in s2_papermeta_json:
            if x['type'] == "node" and x['id'] not in existing_node_ids:
                s2_relatedmeta_json.append(x)
            elif x['type'] == "relationship" and (x['startNodeId'], x['endNodeId']) not in existing_edge_ids:
                s2_relatedmeta_json.append(x)

        target_paper_doi = filtered_s2_paper_metadata[0].get('doi') if isinstance(filtered_s2_paper_metadata, list) and len(filtered_s2_paper_metadata) > 0 else None
        if target_paper_doi is not None:
            if topic:
                topic_hash_id = generate_hash_key(topic)
                if topic_hash_id not in existing_node_ids:
                    topic_node = {
                        'type': 'node',
                        'id': topic_hash_id,
                        'labels': ['Topic'],
                        'properties': {'topic_hash_id': topic_hash_id, 'topic_name': topic, 'hash_method': 'hashlib.sha256'}}
                    s2_relatedmeta_json.append(topic_node)

                if (target_paper_doi, topic_hash_id) not in existing_edge_ids:
                    paper_topic_relationship = {
                        "type": "relationship",
                        "relationshipType": "DISCUSS",
                        "startNodeId": target_paper_doi,
                        "endNodeId": topic_hash_id,
                        "properties": {}}
                    s2_relatedmeta_json.append(paper_topic_relationship)
                    
    return s2_relatedmeta_json