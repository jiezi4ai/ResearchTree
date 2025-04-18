import re
from typing import Optional, List, Dict, Literal

from utils.data_process import generate_hash_key, remove_kth_element, remove_key_values, filter_and_reorder_dict

def reset_id(s2_paper_metadata: List[Dict]|Dict):
    """reset id for paper metadata (from semantic scholar)"""
    if isinstance(s2_paper_metadata, dict):
        s2_paper_metadata = [s2_paper_metadata]

    s2_paper_metadata_updt = []
    for item in s2_paper_metadata:
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
                if year is not None:
                    item['publicationDate'] = f"{year}-01-01"
                else:
                    item['publicationDate'] = '2000-01-01'

            # cut down author information
            item['authors'] = item.get('authors', [])[0:10]

            s2_paper_metadata_updt.append(item)
    return s2_paper_metadata_updt
            

def reset_and_filter_paper(
    s2_paper_metadata: List[Dict]|Dict,
    from_dt: Optional[str] = None,   # filter publish dt no earlier than
    to_dt: Optional[str] = None,   # filter publish dt no late than
    fields_of_study: Optional[List[str]] = None,  # list of field of study
    min_citation_cnt: Optional[int] = 0,  # citation count no less than
    institutions: Optional[List[str]] = None,  # restrcted to list of institutions, to be implemented
    journals: Optional[List[str]] = None,  # restrcted to list of journals, to be implemented
    author_ids: Optional[List[str]] = None,  # restrcted to list of authors' ids
):
    """filter paper metadata (from semantic scholar) based on given criteria"""
    if isinstance(s2_paper_metadata, dict):
        s2_paper_metadata = [s2_paper_metadata]

    s2_paper_metadata_reset = reset_id(s2_paper_metadata)

    s2_paper_metadata_updt = []
    for item in s2_paper_metadata_reset:
        paper_doi = item.get('doi')
        if paper_doi is not None:
            publish_dt = item.get('publicationDate')
            paper_author_ids = [x.get('authorId') for x in item.get('authors') if x.get('authorId') is not None] 
            paper_fields = item.get('fieldsOfStudy', [])
            paper_citation_cnt = item.get('referenceCount', 0)
            
            if from_dt and to_dt and (publish_dt < from_dt or publish_dt > to_dt):  # exclude paper out of time scope
                flag = 0
            elif fields_of_study is not None and paper_fields is not None and len(set(fields_of_study).intersection(set(paper_fields))) == 0:  # exclude paper not in fields of study
                flag = 0
            elif min_citation_cnt and paper_citation_cnt < paper_citation_cnt:   # exclude paper not meeting citation criteria
                flag = 0
            elif author_ids and len(set(paper_author_ids).intersection(set(author_ids))) == 0:  # exclude paper not in author list
                flag = 0
            else:
                flag = 1
        else:
            flag = 0

        if flag == 1:
            s2_paper_metadata_updt.append(item)

    return s2_paper_metadata_updt


def process_paper_metadata(filtered_paper_metadata: List[Dict]|Dict):
    """standardize paper metadata to better suit neo4j format
    Argss:
        filtered_paper_metadata ([List[Dict]|Dict]): paper metadata that has been reset and filtered
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
    if isinstance(filtered_paper_metadata, dict):
        filtered_paper_metadata = [filtered_paper_metadata]

    # then process to standard json format
    s2_papermeta_json = []
    for item in filtered_paper_metadata:
        existing_node_ids = [x['id'] for x in s2_papermeta_json if x['type']=='node']
        existing_edge_ids = [(x['startNodeId'],x['endNodeId']) for x in s2_papermeta_json if x['type']=='relationship']

        paper_doi = item.get('doi') 

        if paper_doi is not None:
            authors = item.get('authors', [])[:10] if item.get('authors', []) is not None else []
            journal = item.get('journal', {}) if item.get('journal', {}) is not None else {}
            venue = item.get('publicationVenue', {}) if item.get('publicationVenue', {}) is not None else {}
            journal_name = journal.get('name') if isinstance(journal, dict) else None
            venue_id = venue.get('id') if isinstance(venue, dict) else None

            # process paper node
            paper_node = {
                "type": "node",
                "id": paper_doi,
                "labels": ["Paper"],
                "properties": item
                }
            s2_papermeta_json.append(paper_node)

            for idx, author in enumerate(authors[:10]):
                # process author node
                author_id = author.get('authorId')
                if author_id is not None:
                    if author_id not in existing_node_ids:
                        author_node = {
                            "type": "node",
                            "id": author.get('authorId'),
                            "labels": ["Author"],
                            "properties": author}
                        s2_papermeta_json.append(author_node)
                
                    # process author -> WRITES -> paper
                    author_order = idx + 1
                    coauthors = remove_kth_element(authors, idx)
                    if (author_id, paper_doi) not in existing_edge_ids:
                        author_paper_relationship = {
                            "type": "relationship",
                            "relationshipType": "WRITES",
                            "startNodeId": author_id,
                            "endNodeId": paper_doi,
                            "properties": {'authorOrder': author_order, 'coauthors': coauthors}
                            }
                        s2_papermeta_json.append(author_paper_relationship)

            # process journal node
            if journal_name is not None:
                journal_hash_id = generate_hash_key(journal_name)
                if journal_hash_id not in existing_node_ids:
                    journal_node = {
                        "type": "node",
                        "id": journal_hash_id,
                        "labels": ["Journal"],
                        "properties": {"jounal_hash_id": journal_hash_id, "name": journal_name, "hash_method":"hashlib.sha256"}}
                    s2_papermeta_json.append(journal_node)
                
                # process paper -> PRINTS_ON -> journal
                if (paper_doi, journal_hash_id) not in existing_edge_ids:
                    if 'arxiv' not in journal_name.lower():  # journal可能会有大量热点，预先进行排除
                        paper_journal_relationship = {
                        "type": "relationship",
                        "relationshipType": "PRINTS_ON",
                        "startNodeId": paper_doi,
                        "endNodeId": journal_hash_id,
                        "properties": journal}
                        s2_papermeta_json.append(paper_journal_relationship)

            # process venue node
            if venue_id is not None:
                if venue_id not in existing_node_ids:
                    venue_node = {
                        "type": "node",
                        "id": venue_id,
                        "labels": ["Venue"],
                        "properties": venue
                        }
                    s2_papermeta_json.append(venue_node)
                
                # process paper -> RELEASES_IN -> venue
                if (paper_doi, venue_id) not in existing_edge_ids:
                    if 'arxiv' not in venue.get('name').lower():  # venue可能会有大量热点，预先进行排除
                        paper_venue_relationship = {
                        "type": "relationship",
                        "relationshipType": "RELEASES_IN",
                        "startNodeId": paper_doi,
                        "endNodeId": venue_id,
                        "properties": {}}
                        s2_papermeta_json.append(paper_venue_relationship)
    return s2_papermeta_json


def process_author_metadata(
        s2_author_metadata: List[Dict]|Dict,
        from_dt: Optional[str] = None,   # filter publish dt no earlier than
        to_dt: Optional[str] = None,   # filter publish dt no late than
        fields_of_study: Optional[List[str]] = None,  # list of field of study
        min_citation_cnt: Optional[int] = 0,  # citation count no less than
        institutions: Optional[List[str]] = None,  # restrcted to list of institutions, to be implemented
        journals: Optional[List[str]] = None,  # restrcted to list of journals, to be implemented
        author_ids: Optional[List[str]] = None,  # restrcted to list of authors' ids      
    ):
    """standardize author metadata to better suit neo4j format
    Argss:
        s2_author_metadata ([List[Dict]|Dict]): author metadata
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
    if isinstance(s2_author_metadata, dict):
        s2_author_metadata = [s2_author_metadata]

    s2_authormeta_json = []
    for item in s2_author_metadata:
        existing_node_ids = [x['id'] for x in s2_authormeta_json if x['type']=='node']
        existing_edge_ids = [(x['startNodeId'],x['endNodeId'])  for x in s2_authormeta_json if x['type']=='relationship']

        author_id = item.get('authorId')
        if author_id is not None:
            papers_metadata = item.get('papers', [])

            # process author metadata
            author_node = {
                        "type": "node",
                        "id": author_id,
                        "labels": ["Author"],
                        "properties": remove_key_values(item, 'papers')}
            s2_authormeta_json.append(author_node)    

            # process institution metadata
            institutions = item.get('affiliations', [])
            if isinstance(institutions, list) and len(institutions) > 0:
                for inst in institutions:
                    inst_hash_id = generate_hash_key(inst)
                    if inst_hash_id not in existing_node_ids:
                        inst_node = {
                                    "type": "node",
                                    "id": inst_hash_id,
                                    "labels": ["Affiliation"],
                                    "properties":  {'affiliation_hash_id': inst_hash_id, 'affiliation_name': inst, 'hash_method': 'hashlib.sha256'}}
                        s2_authormeta_json.append(inst_node)    

                    if (author_id, inst_hash_id) not in existing_edge_ids:
                        author_inst_relationship = {
                            "type": "relationship",
                            "relationshipType": "WORKS_IN",
                            "startNodeId": author_id,
                            "endNodeId": inst_hash_id,
                            "properties": {}}
                        s2_authormeta_json.append(author_inst_relationship)

            # process all paper metadata
            if isinstance(papers_metadata, list) is not None and len(papers_metadata) > 0:
                filtered_s2_paper_metadata = reset_and_filter_paper(
                            s2_paper_metadata=papers_metadata,
                            from_dt=from_dt, 
                            to_dt=to_dt, 
                            fields_of_study=fields_of_study)
                s2_papermeta_json = process_paper_metadata(filtered_s2_paper_metadata)
                for x in s2_papermeta_json:
                    if x['type'] == "node" and x['id'] not in existing_node_ids:
                        s2_authormeta_json.append(x)
                    elif x['type'] == "relationship" and (x['startNodeId'], x['endNodeId']) not in existing_edge_ids:
                        s2_authormeta_json.append(x)
    return s2_authormeta_json


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