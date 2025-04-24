# s2_data_process.py
# for now only filter over paper

import re
from typing import Optional, List, Dict, Literal

from utils.data_process import generate_hash_key, remove_kth_element, remove_key_values, filter_and_reorder_dict

def align_paper_metadata(s2_papers_metadata: List[Dict]|Dict):
    """reset id and keys for paper metadata"""
    if isinstance(s2_papers_metadata, dict):
        s2_papers_metadata = [s2_papers_metadata]

    s2_papers_dict = []
    for item in s2_papers_metadata:
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
            

def filter_papers(
    aligned_s2_papers: List[Dict]|Dict,  # paper data already process by align_paper_metadata
    from_dt: Optional[str] = None,   # filter publish dt no earlier than
    to_dt: Optional[str] = None,   # filter publish dt no late than
    fields_of_study: Optional[List[str]] = None,  # list of field of study
    min_citation_cnt: Optional[int] = 0,  # citation count no less than
    author_ids: Optional[List[str]] = None,  # restrcted to list of authors' ids
    # institutions: Optional[List[str]] = None,  # restrcted to list of institutions, to be implemented
    # journals: Optional[List[str]] = None,  # restrcted to list of journals, to be implemented
):
    """filter paper metadata (from semantic scholar) based on given criteria"""
    review_result = {
        'off_dt_range': set(),
        'off_fields_of_study': set(),
        'below_min_citation': set(),
        'off_author_scope': set(),
    }

    filter_fos_set = set(fields_of_study) if fields_of_study is not None else None
    filter_author_id_set = set(author_ids) if author_ids is not None else None

    for item in aligned_s2_papers:
        paper_doi = item.get('doi')
        if paper_doi is None:
            continue

        publish_dt = item.get('publicationDate')
        paper_author_ids_set = set(x.get('authorId') for x in item.get('authors', []) if x.get('authorId') is not None)
        paper_fos_set = set(item.get('fieldsOfStudy', []))
        paper_citation_cnt = item.get('referenceCount', 0)
        
        # --- Filtering Logic ---
        # exclude paper out of time scope
        if from_dt is not None and to_dt  is not None and (publish_dt < from_dt or publish_dt > to_dt):  
            review_result['off_dt_range'].add(paper_doi)
        
        # exclude paper not in fields of study
        if filter_fos_set is not None and paper_fos_set is not None and not filter_fos_set.intersection(paper_fos_set): 
            review_result['off_fields_of_study'].add(paper_doi)

        # exclude paper not meeting citation criteria
        if min_citation_cnt is not None and paper_citation_cnt < min_citation_cnt:   
            review_result['below_min_citation'].add(paper_doi)
        
        # exclude paper not in author list
        if filter_author_id_set is not None and not filter_author_id_set.intersection(paper_author_ids_set):
            review_result['off_author_scope'].add(paper_doi)

    return review_result


def process_paper_data(
        s2_papers_metadata: List[Dict]|Dict,
        from_dt: Optional[str] = None,   # filter publish dt no earlier than
        to_dt: Optional[str] = None,   # filter publish dt no late than
        fields_of_study: Optional[List[str]] = None,  # list of field of study
        ):
    """standardize paper metadata to better suit neo4j format
    Argss:
        s2_papers_metadata ([List[Dict]|Dict]): paper metadata that has been reset and filtered
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
    # align ids and keys for paper metadata
    aligned_papers_dict = align_paper_metadata(s2_papers_metadata)

    # get paper filter results
    exclusion_info = filter_papers(aligned_papers_dict, from_dt, to_dt, fields_of_study)
    exc_dois = (exclusion_info['off_dt_range'] | exclusion_info['off_fields_of_study'] |
                exclusion_info['below_min_citation'] | exclusion_info['off_author_scope'])

    # Process into standard JSON format, separating included/excluded
    inc_json, exc_json = [], []
    # Use separate sets for deduplication within each list
    inc_node_ids, exc_node_ids = set(), set()
    inc_edge_tuples, exc_edge_tuples = set(), set()
    
    # Define paper properties to include (excluding 'authors')
    paper_props = ['doi', 'title', 'abstract', 'year', 'publicationDate',
                    'citationCount', 'referenceCount', 'influentialCitationCount', 'arxivId', 'arxivUrl',
                    'isOpenAccess', 'openAccessPdf', 'version', 'paperId',
                    'fieldsOfStudy'] # Add/remove as needed

    for item in aligned_papers_dict:
        paper_doi = item.get('doi') 
        if paper_doi is None:
            continue 

        if paper_doi in exc_dois:
            s2_paper_json = exc_json
            processed_node_ids = exc_node_ids
            processed_edge_tuples = exc_edge_tuples
        else:
            s2_paper_json = inc_json
            processed_node_ids = inc_node_ids
            processed_edge_tuples = inc_edge_tuples

        # --- Process Paper Node ---

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
                    author_props = {k: v for k, v in author.items() if k in ['authorId', 'name'] and v is not None}
                    author_node = {
                        "type": "node",
                        "id": author_id,
                        "labels": ["Author"],
                        "properties": author_props}
                    s2_paper_json.append(author_node)
                    processed_node_ids.add(author_id)
            
                # process author -> WRITES -> paper
                edge_tuple = (author_id, paper_doi, "WRITES")
                if edge_tuple not in processed_edge_tuples:
                    author_order = idx + 1
                    author_paper_relationship = {
                        "type": "relationship",
                        "relationshipType": "WRITES",
                        "startNodeId": author_id,
                        "endNodeId": paper_doi,
                        "properties": {'authorOrder': author_order}
                        }
                    s2_paper_json.append(author_paper_relationship)
                    processed_edge_tuples.add(edge_tuple)

        # --- Process Journal ---
        journal = item.get('journal', {})

        paper_in_journal_props = {}
        if isinstance(journal, dict):
            journal_name = journal.get('name')  
            paper_in_journal_props['volume'] = journal.get('volume')
            paper_in_journal_props['pages'] = journal.get('pages')
            # Filter None values from properties
            paper_in_journal_props = {k:v for k,v in paper_in_journal_props.items() if v is not None}
        else:
            journal_name = None

        if journal_name is not None:
            journal_hash_id = generate_hash_key(journal_name)
            if journal_hash_id not in processed_node_ids:
                journal_props = {"journal_hash_id": journal_hash_id, "name": journal_name, "hash_method":"hashlib.sha256"}
                journal_node = {
                    "type": "node",
                    "id": journal_hash_id,
                    "labels": ["Journal"],
                    "properties": journal_props}
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
                        "properties": paper_in_journal_props}
                    s2_paper_json.append(paper_journal_relationship)
                    processed_edge_tuples.add(edge_tuple)

        # --- Process Venue ---
        venue = item.get('publicationVenue', {})
        venue_id = venue.get('id') if isinstance(venue, dict) else None
        venue_name = venue.get('name', '') if isinstance(venue, dict) else ''
        if venue_id is not None:
            if venue_id not in processed_node_ids:
                venue_props = {k: v for k, v in venue.items() if k in ['id', 'name', 'type', 'url'] and v is not None}
                venue_node = {
                    "type": "node",
                    "id": venue_id,
                    "labels": ["Venue"],
                    "properties": venue_props
                    }
                s2_paper_json.append(venue_node)
                processed_node_ids.add(venue_id)
            
            # process paper -> RELEASES_IN -> venue
            edge_tuple = (paper_doi, venue_id, "RELEASES_IN")
            if edge_tuple not in processed_edge_tuples:
                if 'arxiv' not in venue_name.lower():  # exclude arxiv from venue
                    paper_venue_relationship = {
                        "type": "relationship",
                        "relationshipType": "RELEASES_IN",
                        "startNodeId": paper_doi,
                        "endNodeId": venue_id,
                        "properties": {}}
                    s2_paper_json.append(paper_venue_relationship)
                    processed_edge_tuples.add(edge_tuple)

    processed_result = {'include': inc_json,
                        'exclude': exc_json,
                        'exclusion_info':exclusion_info}
    return processed_result


def process_author_data(
        s2_authors: List[Dict]|Dict,
        from_dt: Optional[str] = None,   # filter publish dt no earlier than
        to_dt: Optional[str] = None,   # filter publish dt no late than
        fields_of_study: Optional[List[str]] = None,  # list of field of study   
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

    # Final accumulators for the entire batch of authors
    final_inc_json, final_exc_json = [], []
    final_inc_node_ids, final_exc_node_ids = set(), set()
    final_inc_edge_tuples, final_exc_edge_tuples = set(), set()

    # Accumulator for exclusion reasons across all processed papers for all authors
    aggregated_exclusion_info = {
        'off_dt_range': set(),
        'off_fields_of_study': set(),
        'below_min_citation': set(),
        'off_author_scope': set(), 
    }

    # --- Define properties to keep for Author nodes ---
    author_props_to_keep = [
        'authorId', 'name', 'aliases', 'url', 'hIndex',
        'paperCount', 'citationCount', # Add other relevant author fields
    ]
    # --- Define properties to keep for Institution nodes ---
    inst_props_to_keep = ['institution_hash_id', 'name', 'hash_method']

    for author_item in s2_authors:
        author_id = author_item.get('authorId')
        if author_id is None:
            continue

        # --- 1. Process Author Node ---
        # Add Author node to the include list if not already present
        author_props = {
                k: v for k, v in author_item.items()
                if k in author_props_to_keep and v is not None
            }
        # Ensure essential props are present
        if 'authorId' not in author_props: author_props['authorId'] = author_id
        if 'name' not in author_props: author_props['name'] = author_item.get('name', 'Unknown Name') # Add default if missing
        author_node = {
                    "type": "node",
                    "id": author_id,
                    "labels": ["Author"],
                    "properties": author_props}
        final_inc_json.append(author_node)
        final_inc_node_ids.add(author_id) 

        # --- 2. Process Affiliations (Institutions and WORKS_IN) ---
        # process institution metadata
        institutions = author_item.get('affiliations', [])
        if isinstance(institutions, list):
            for inst_data in institutions:
                inst_name = None
                if isinstance(inst_data, str):
                    inst_name = inst_data
                elif isinstance(inst_data, dict):
                    inst_name = inst_data.get('name', inst_data.get('institution'))

                if inst_name and inst_name.strip(): # Check if name is valid
                    inst_hash_id = generate_hash_key(inst_name)
                    if inst_hash_id not in final_inc_node_ids:
                        inst_props = {
                            'institution_hash_id': inst_hash_id,
                            'name': inst_name,
                            'hash_method': 'hashlib.sha256'
                            }
                        inst_node = {
                            "type": "node",
                            "id": inst_hash_id,
                            "labels": ["Institution"],
                            "properties": inst_props}
                        final_inc_json.append(inst_node)
                        final_inc_node_ids.add(inst_hash_id)

                    # process author -> WORKS_IN -> affiliations
                    edge_tuple = (author_id, inst_hash_id, "WORKS_IN")
                    if edge_tuple not in final_inc_edge_tuples:
                        author_inst_relationship = {
                            "type": "relationship",
                            "relationshipType": "WORKS_IN",
                            "startNodeId": author_id,
                            "endNodeId": inst_hash_id,
                            "properties": {} 
                        }
                        final_inc_json.append(author_inst_relationship)
                        final_inc_edge_tuples.add(edge_tuple)

        # --- 3. Process Associated Papers ---
        s2_papers = author_item.get('papers', [])
        if isinstance(s2_papers, list) and len(s2_papers) > 0:
            papers_processed = process_paper_data(s2_papers, from_dt, to_dt, fields_of_study)
            for item in papers_processed['include']:
                if item['type'] == 'node' and item['id'] not in final_inc_node_ids:
                    final_inc_json.append(item)
                    final_inc_node_ids.add(item['id'])
                elif item['type'] == 'relationship':
                    edge_tuple = (item['startNodeId'], item['endNodeId'], item['relationshipType'])
                    if edge_tuple not in final_inc_edge_tuples:
                        final_inc_json.append(item)
                        final_inc_edge_tuples.add(edge_tuple)
            
            for item in papers_processed['exclude']:
                if item['type'] == 'node' and item['id'] not in final_exc_node_ids:
                    final_exc_json.append(item)
                    final_exc_node_ids.add(item['id'])
                elif item['type'] == 'relationship':
                    edge_tuple = (item['startNodeId'], item['endNodeId'], item['relationshipType'])
                    if edge_tuple not in final_exc_edge_tuples:
                        final_exc_json.append(item)
                        final_exc_edge_tuples.add(edge_tuple)

            # Aggregate exclusion information from this batch of papers
            current_exclusion_reasons = papers_processed['exclusion_info']
            for reason, doi_set in current_exclusion_reasons.items():
                if reason in aggregated_exclusion_info:
                    aggregated_exclusion_info[reason].update(doi_set)

    processed_result = {
        'include': final_inc_json,
        'exclude': final_exc_json,
        'exclusion_info': aggregated_exclusion_info
    }

    return processed_result


def process_citation_data(
        original_paper_doi: str,
        s2_citations: List[Dict]|Dict,
        citation_type: Literal['citingPaper','citedPaper'],
        from_dt: Optional[str] = None,   # filter publish dt no earlier than
        to_dt: Optional[str] = None,   # filter publish dt no late than
        fields_of_study: Optional[List[str]] = None,  # list of field of study    
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
    # for citations (citing or cited papers)
    if isinstance(s2_citations, dict):
        s2_citations = [s2_citations] 

    # Accumulators for the final result
    final_inc_json, final_exc_json = [], []
    final_inc_node_ids, final_exc_node_ids = set(), set()
    final_inc_edge_tuples, final_exc_edge_tuples = set(), set()

    # Accumulator for exclusion reasons across all processed citations
    aggregated_exclusion_info = {
        'off_dt_range': set(),
        'off_fields_of_study': set(),
        'below_min_citation': set(),
        'off_author_scope': set(),
    }

    # Key for citation properties within the citation record item
    citation_props_keys = ['isInfluential', 'contexts', 'intents', 'contextsWithIntent']

    for citation_item in s2_citations:
        # 1. Extract raw metadata for the citation paper
        s2_papers = citation_item.get(citation_type)  # get either citing or cited paper metadata
        if not isinstance(s2_papers, dict):
            continue

        # 2. Align the citation paper's metadata to get its canonical DOI
        aligned_list = align_paper_metadata(s2_papers)
        if not aligned_list:
            continue
        citation_paper_aligned = aligned_list[0]
        citation_paper_doi = citation_paper_aligned.get('doi')
        if citation_paper_doi is None:
            continue
        
        # Avoid processing self-citations directly (can happen in API results)
        if citation_paper_doi == original_paper_doi:
            continue
    
        # 3. Process the citation paper (using its raw metadata as process_paper_data handles alignment)
        #    This applies filters and generates Neo4j JSON for this single paper.
        papers_processed = process_paper_data(s2_papers, from_dt, to_dt, fields_of_study)

        # 4. Determine if the *other* paper was excluded by the filters
        #    Check if its DOI appears in any exclusion set from its processing run.
        current_exclusion_reasons = papers_processed['exclusion_info']
        is_excluded = any(citation_paper_doi in reason_set for reason_set in current_exclusion_reasons.values())

        # 5. Merge results and potentially create CITES relationship
        if not is_excluded:
            # Merge included data into final results, handling duplicates
            for item in papers_processed['include']:
                if item['type'] == 'node':
                    if item['id'] not in final_inc_node_ids:
                        final_inc_json.append(item)
                        final_inc_node_ids.add(item['id'])
                elif item['type'] == 'relationship':
                    edge_tuple = (item['startNodeId'], item['endNodeId'], item['relationshipType'])
                    if edge_tuple not in final_inc_edge_tuples:
                        final_inc_json.append(item)
                        final_inc_edge_tuples.add(edge_tuple)

            # Create the CITES relationship since the target paper is included
            if citation_type == 'citedPaper':  # original paper -> CITES -> other paper
                start_node_id = original_paper_doi
                end_node_id = citation_paper_doi
            else:  # other paper -> CITES -> original paper
                start_node_id = citation_paper_doi
                end_node_id = original_paper_doi

            # append paper -> CITES -> paper relationship
            edge_tuple = (start_node_id, end_node_id, 'CITES')
            if edge_tuple not in final_inc_edge_tuples:
                cites_properties = filter_and_reorder_dict(citation_item, citation_props_keys)
                paper_cites_relationship = {
                    "type": "relationship",
                    "relationshipType": "CITES",
                    "startNodeId": start_node_id,
                    "endNodeId": end_node_id,
                    "properties": cites_properties}
                final_inc_json.append(paper_cites_relationship)
                final_inc_edge_tuples.add(edge_tuple)
        else:
            # Merge excluded data into final results, handling duplicates
            for item in papers_processed['exclude']:
                if item['type'] == 'node':
                    if item['id'] not in final_exc_node_ids:
                        final_exc_json.append(item)
                        final_exc_node_ids.add(item['id'])
                elif item['type'] == 'relationship':
                    edge_tuple = (item['startNodeId'], item['endNodeId'], item['relationshipType'])
                    if edge_tuple not in final_exc_edge_tuples:
                        final_exc_json.append(item)
                        final_exc_edge_tuples.add(edge_tuple)

        # 6. Aggregate exclusion information regardless of include/exclude status
        for reason, doi_set in current_exclusion_reasons.items():
            if reason in aggregated_exclusion_info:
                aggregated_exclusion_info[reason].update(doi_set)

    # 7. Assemble final result
    processed_result = {
        'include': final_inc_json,
        'exclude': final_exc_json,
        'exclusion_info': aggregated_exclusion_info
    }

    return processed_result
    

def process_related_metadata(
        s2_related_papers: List[Dict]|Dict,
        topic: Optional[str] = None,
        from_dt: Optional[str] = None,   # filter publish dt no earlier than
        to_dt: Optional[str] = None,   # filter publish dt no late than
        fields_of_study: Optional[List[str]] = None,  # list of field of study    
    ):
    """standardize paper citation relationships to better suit neo4j format """
    if isinstance(s2_related_papers, dict):
        s2_related_papers = [s2_related_papers]

    # Final accumulators for the entire batch of authors
    final_inc_json, final_exc_json = [], []
    final_inc_node_ids, final_exc_node_ids = set(), set()
    final_inc_edge_tuples, final_exc_edge_tuples = set(), set()

    # Accumulator for exclusion reasons across all processed papers for all authors
    aggregated_exclusion_info = {
        'off_dt_range': set(),
        'off_fields_of_study': set(),
        'below_min_citation': set(),
        'off_author_scope': set(), # Note: 'off_author_scope' might be less relevant here
                                   # unless filtering papers based on co-authors.
    }
    
    for paper in s2_related_papers:
        # 1. Extract raw metadata for the related paper
        if not isinstance(paper, dict):
            continue

        # 2. Align the related paper's metadata to get its canonical DOI
        aligned_list = align_paper_metadata(paper)
        if not aligned_list:
            continue
        paper_aligned = aligned_list[0]
        paper_doi = paper_aligned.get('doi')
        if paper_doi is None:
            continue

        # 3. add topic node and paper -> DISCUSS -> topic relationship
        if paper_doi is not None:
            if topic:
                topic_hash_id = generate_hash_key(topic)
                if topic_hash_id not in final_inc_node_ids:
                    topic_props = {'topic_hash_id': topic_hash_id, 'topic_name': topic, 'hash_method': 'hashlib.sha256'}
                    topic_node = {
                        'type': 'node',
                        'id': topic_hash_id,
                        'labels': ['Topic'],
                        'properties': topic_props}
                    final_inc_json.append(topic_node)
                    final_inc_node_ids.add(topic_hash_id)

                edge_tuple = (paper_doi, topic_hash_id, "DISCUSS")
                if edge_tuple not in final_inc_edge_tuples:
                    paper_topic_relationship = {
                        "type": "relationship",
                        "relationshipType": "DISCUSS",
                        "startNodeId": paper_doi,
                        "endNodeId": topic_hash_id,
                        "properties": {}}
                    final_inc_json.append(paper_topic_relationship)
                    final_exc_edge_tuples.add(edge_tuple)

        # 3. Process related paper and convert to neo4j format json.
        papers_processed = process_paper_data(paper, from_dt, to_dt, fields_of_study)
        for item in papers_processed['include']:
            if item['type'] == 'node' and item['id'] not in final_inc_node_ids:
                final_inc_json.append(item)
                final_inc_node_ids.add(item['id'])
            elif item['type'] == 'relationship':
                edge_tuple = (item['startNodeId'], item['endNodeId'], item['relationshipType'])
                if edge_tuple not in final_inc_edge_tuples:
                    final_inc_json.append(item)
                    final_inc_edge_tuples.add(edge_tuple)
        
        for item in papers_processed['exclude']:
            if item['type'] == 'node' and item['id'] not in final_exc_node_ids:
                final_exc_json.append(item)
                final_exc_node_ids.add(item['id'])
            elif item['type'] == 'relationship':
                edge_tuple = (item['startNodeId'], item['endNodeId'], item['relationshipType'])
                if edge_tuple not in final_exc_edge_tuples:
                    final_exc_json.append(item)
                    final_exc_edge_tuples.add(edge_tuple)

        # Aggregate exclusion information from this batch of papers
        current_exclusion_reasons = papers_processed['exclusion_info']
        for reason, doi_set in current_exclusion_reasons.items():
            if reason in aggregated_exclusion_info:
                aggregated_exclusion_info[reason].update(doi_set)

    # --- 4. Assemble final result ---
    processed_result = {
        'include': final_inc_json,
        'exclude': final_exc_json,
        'exclusion_info': aggregated_exclusion_info
    }

    return processed_result