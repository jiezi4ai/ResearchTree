import re
from typing import Optional, List, Dict, Literal

from utils.data_process import rename_key_in_dict, remove_kth_element, remove_key_values, filter_and_reorder_dict

def reset_id(s2_paper_metadata: List[Dict]|Dict):
    """reset id for paper metadata (from semantic scholar)"""
    if type(origial_paper_metadata) == dict:
        origial_paper_metadata = [origial_paper_metadata]

    s2_paper_metadata_updt = []
    for item in s2_paper_metadata:
        # set up paper id
        paper_id = item.get('paperId')
        if paper_id is not None:
            arxiv_id = item.get('externalIds',{}).get('ArXiv')  # arxiv id
            if arxiv_id is not None:
                arxiv_no = arxiv_id.replace('10.48550/arXiv.', '') 
                arxiv_id = re.sub(r'v\d+$', '', arxiv_no)
                version_match = re.search(r'v\d+$', arxiv_no)
                # generate arxiv related info
                item['version'] = version_match.group(0) if version_match else ""
                item['arxivUrl'] = f"https://arxiv.org/abs/{arxiv_no}"
                item['isOpenAccess'] = True
                item['openAccessPdf'] = f"https://arxiv.org/pdf/{arxiv_no}"
            item['arxivId'] = arxiv_id

            doi = item.get('externalIds',{}).get('DOI')  # doi
            if doi is None and arxiv_id is not None:
                doi = f"10.48550/arXiv.{arxiv_id}"  # assign 10.48550/arXiv. for arxiv id https://info.arxiv.org/help/doi.html
            item['DOI'] = doi

            # for unique id
            if arxiv_id is not None:
                item['id'] = f"10.48550/arXiv.{arxiv_id}"
            elif doi is not None:
                item['id'] = doi
            else:
                item['id'] = paper_id
            
            s2_paper_metadata_updt.append(item)
    return s2_paper_metadata_updt
            

def filter_by_condition(
    s2_paper_metadata: List[Dict]|Dict,
    from_dt: Optional[str] = None,   # filter publish dt no earlier than
    to_dt: Optional[str] = None,   # filter publish dt no late than
    field: Optional[List[str]] = None,  # list of field of study
    min_citation_cnt: Optional[int] = 0,  # citation count no less than
    institutions: Optional[List[str]] = None,  # restrcted to list of institutions, to be implemented
    journals: Optional[List[str]] = None,  # restrcted to list of journals, to be implemented
    author_ids: Optional[List[str]] = None,  # restrcted to list of authors' ids
):
    """filter paper metadata (from semantic scholar) based on given criteria"""
    if type(s2_paper_metadata) == dict:
        s2_paper_metadata = [s2_paper_metadata]

    s2_paper_metadata_updt = []
    for item in s2_paper_metadata:
        paper_id = item.get('paperId')
        if paper_id is not None:
            publish_dt = item.get('publicationDate')
            year = item.get('year')
            if publish_dt is None:
                if year is not None:
                    publish_dt = f"{year}-01-01"
                else:
                    publish_dt = '2000-01-01'
        
        paper_author_ids = [x.get('authorId') for x in item.get('authors') if x.get('authorId') is not None] 
        paper_fields = item.get('fieldsOfStudy', [])
        paper_citation_cnt = item.get('referenceCount', 0)

        flag = 1
        if from_dt and to_dt and (publish_dt < from_dt or publish_dt > to_dt):
            flag = 0
        
        if field and len(set(paper_fields).intersection(set(paper_fields))) == 0:
            flag = 0

        if min_citation_cnt and paper_citation_cnt < paper_citation_cnt:
            flag = 0
        
        if author_ids and len(set(paper_author_ids).intersection(set(author_ids))) == 0:
            flag = 0

        if flag == 1:
            s2_paper_metadata_updt.append(item)

    return s2_paper_metadata_updt


def process_paper_metadata(
        s2_paper_metadata: List[Dict]|Dict,
        from_dt: Optional[str] = None,   # filter publish dt no earlier than
        to_dt: Optional[str] = None,   # filter publish dt no late than
        field: Optional[List[str]] = None,  # list of field of study
        min_citation_cnt: Optional[int] = 0,  # citation count no less than
        institutions: Optional[List[str]] = None,  # restrcted to list of institutions, to be implemented
        journals: Optional[List[str]] = None,  # restrcted to list of journals, to be implemented
        author_ids: Optional[List[str]] = None,  # restrcted to list of authors' ids        
        ):
    """standardize paper metadata to better suit neo4j format
    Argss:
        s2_paper_metadata ([List[Dict]|Dict]): paper metadata
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
    if type(s2_paper_metadata) == dict:
        s2_paper_metadata = [s2_paper_metadata]
    
    # reset id for paper metadata
    s2_paper_metadata = reset_id(s2_paper_metadata)

    # filter by conditions
    s2_paper_metadata = filter_by_condition(
        s2_paper_metadata,
        from_dt,
        to_dt,
        field,
        min_citation_cnt,
        institutions,
        journals,
        author_ids
    )

    # then process to standard json format
    s2_papermeta_json = []
    for item in s2_paper_metadata:
        item = rename_key_in_dict(item, {'url': 's2Url'})
        item = rename_key_in_dict(item, {'paperId', 's2PaperId'})
        paper_id = item.get('s2PaperId')  # semantic scholar paper id

        if paper_id is not None:
            id = item.get('id')
            authors = item.get('authors', [])[:10] if item.get('authors', []) is not None else []
            journal = item.get('journal', {}) if item.get('journal', {}) is not None else {}
            venue = item.get('publicationVenue', {}) if item.get('publicationVenue', {}) is not None else {}
            journal_name = journal.get('name')
            venue_id = venue.get('id')

            # process paper node
            paper_node = {
                "type": "node",
                "id": id,
                "labels": ["Paper"],
                "properties": item
                }
            s2_papermeta_json.append(paper_node)


            for idx, author in enumerate(authors[:10]):
                # process author node
                author_id = author.get('authorId')
                if author_id is not None:
                    author_node = {
                        "type": "node",
                        "id": author.get('authorId'),
                        "labels": ["Author"],
                        "properties": author}
                    s2_papermeta_json.append(author_node)
                
                    # process author -> WRITES -> paper
                    author_order = idx + 1
                    coauthors = remove_kth_element(authors, idx)
                    author_paper_relationship = {
                        "type": "relationship",
                        "relationshipType": "WRITES",
                        "startNodeId": author_id,
                        "endNodeId": id,
                        "properties": {'authorOrder': author_order, 'coauthors': coauthors}
                        }
                    s2_papermeta_json.append(author_paper_relationship)

            # process journal node
            if journal_name is not None:
                journal_node = {
                    "type": "node",
                    "id": journal_name,
                    "labels": ["Journal"],
                    "properties": {"name": journal_name}}
                s2_papermeta_json.append(journal_node)
                
                # process paper -> PRINTS_ON -> journal
                if 'arxiv' not in journal_name.lower():  # journal可能会有大量热点，预先进行排除
                    paper_journal_relationship = {
                    "type": "relationship",
                    "relationshipType": "PRINTS_ON",
                    "startNodeId": id,
                    "endNodeId": journal_name,
                    "properties": journal}
                    s2_papermeta_json.append(paper_journal_relationship)

            # process venue node
            if venue_id is not None:
                venue_node = {
                    "type": "node",
                    "id": venue_id,
                    "labels": ["Venue"],
                    "properties": venue
                    }
                s2_papermeta_json.append(venue_node)
                
                # process paper -> RELEASES_IN -> venue
                if 'arxiv' not in venue.get('name').lower():  # venue可能会有大量热点，预先进行排除
                    paper_venue_relationship = {
                    "type": "relationship",
                    "relationshipType": "RELEASES_IN",
                    "startNodeId": id,
                    "endNodeId": venue_id,
                    "properties": {}}
                    s2_papermeta_json.append(paper_venue_relationship)
        return s2_papermeta_json


def process_author_metadata(
        s2_author_metadata: List[Dict]|Dict,
        from_dt: Optional[str] = None,   # filter publish dt no earlier than
        to_dt: Optional[str] = None,   # filter publish dt no late than
        field: Optional[List[str]] = None,  # list of field of study
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
    if type(s2_author_metadata) == dict:
        s2_author_metadata = [s2_author_metadata]

    s2_authormeta_json = []
    for item in s2_author_metadata:
        item = rename_key_in_dict(item, {'url': 's2Url'})
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

            # process all paper metadata
            if papers_metadata is not None and papers_metadata != []:
                s2_papermeta_json = process_paper_metadata(
                    papers_metadata,
                    from_dt,
                    to_dt,
                    field,
                    min_citation_cnt,
                    institutions,
                    journals,
                    author_ids
                )
                s2_authormeta_json.extend(s2_papermeta_json)

        return s2_authormeta_json


def process_citation_metadata(
        original_paper_metadata: Dict,
        s2_citation_metadata: List[Dict]|Dict,
        type: Literal['citing','cited'],
        from_dt: Optional[str] = None,   # filter publish dt no earlier than
        to_dt: Optional[str] = None,   # filter publish dt no late than
        field: Optional[List[str]] = None,  # list of field of study
        min_citation_cnt: Optional[int] = 0,  # citation count no less than
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

    source_paper_id = original_paper_metadata.get('paperId')
    if source_paper_id is not None:
        # for original paper
        s2_papermeta_json = process_paper_metadata(
            original_paper_metadata,
            from_dt,
            to_dt,
            field,
            min_citation_cnt,
            institutions,
            journals,
            author_ids
        )
        s2_citationmeta_json.extend(s2_papermeta_json)

        # for citations (citing or cited papers)
        if type(s2_citation_metadata) == dict:
            s2_citation_metadata = [s2_citation_metadata]
        
        for item in s2_citation_metadata:
            if type == 'citing':  # source paper citing target papers
                s2_paper_metadata = item.get('citedPaper')
                target_paper_id = s2_paper_metadata.get('paperId')
                if target_paper_id is not None:
                    s2_papermeta_json = process_paper_metadata(
                        original_paper_metadata,
                        from_dt,
                        to_dt,
                        field,
                        min_citation_cnt,
                        institutions,
                        journals,
                        author_ids
                    )
                    s2_citationmeta_json.extend(s2_papermeta_json)
                    start_node_id = source_paper_id
                    end_node_id = s2_papermeta_json[0]['id']

            else:  # source paper cited by target papers
                s2_paper_metadata = item.get('citingPaper')
                target_paper_id = s2_paper_metadata.get('paperId')
                if target_paper_id is not None:
                    s2_papermeta_json = process_paper_metadata(
                        original_paper_metadata,
                        from_dt,
                        to_dt,
                        field,
                        min_citation_cnt,
                        institutions,
                        journals,
                        author_ids
                    )
                    s2_citationmeta_json.extend(s2_papermeta_json)
                    start_node_id = s2_papermeta_json[0]['id']
                    end_node_id = source_paper_id 

            # append relationship
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
        original_paper_metadata: Dict,
        s2_related_metadata: List[Dict]|Dict,
        properties: Dict,
        from_dt: Optional[str] = None,   # filter publish dt no earlier than
        to_dt: Optional[str] = None,   # filter publish dt no late than
        field: Optional[List[str]] = None,  # list of field of study
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
    
    source_paper_id = original_paper_metadata.get('paperId')
    if source_paper_id is not None:
        # for original paper
        s2_papermeta_json = process_paper_metadata(
            original_paper_metadata,
            from_dt,
            to_dt,
            field,
            min_citation_cnt,
            institutions,
            journals,
            author_ids)
        s2_relatedmeta_json.extend(s2_papermeta_json)

        # for related papers
        if type(s2_related_metadata) == dict:
            s2_related_metadata = [s2_related_metadata]
        
        for item in s2_related_metadata:
            target_paper_id = item.get('paperId')
            if target_paper_id is not None:
                s2_papermeta_json = process_paper_metadata(
                    original_paper_metadata,
                    from_dt,
                    to_dt,
                    field,
                    min_citation_cnt,
                    institutions,
                    journals,
                    author_ids)
                s2_relatedmeta_json.extend(s2_papermeta_json)

            # append relationship (bidirection)
            paper_cites_relationship = {
                "type": "relationship",
                "relationshipType": "RELATES",
                "startNodeId": source_paper_id,
                "endNodeId": target_paper_id,
                "properties": properties}
            s2_relatedmeta_json.append(paper_cites_relationship)

            paper_cites_relationship = {
                "type": "relationship",
                "relationshipType": "RELATES",
                "startNodeId": target_paper_id,
                "endNodeId": source_paper_id,
                "properties": properties}
            s2_relatedmeta_json.append(paper_cites_relationship)

        return s2_relatedmeta_json