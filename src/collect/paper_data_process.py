# s2_data_process.py
# for now only filter over paper

import re
import hashlib
from typing import Optional, List, Dict, Literal, Union, Set, Tuple

# --- Constants for Labels and Relationship Types ---
NODE_PAPER = "Paper"
NODE_AUTHOR = "Author"
NODE_JOURNAL = "Journal"
NODE_VENUE = "Venue"
NODE_INSTITUTION = "Institution"
NODE_TOPIC = "Topic"

REL_WRITES = "WRITES"
REL_PRINTS_ON = "PRINTS_ON"
REL_RELEASES_IN = "RELEASES_IN"
REL_WORKS_IN = "WORKS_IN"
REL_CITES = "CITES"
REL_DISCUSS = "DISCUSS"


def generate_hash_key(input_string):
  """Generate hash key using SHA-256 algorithm"""
  encoded_string = input_string.encode('utf-8') 
  hash_object = hashlib.sha256(encoded_string)
  hex_dig = hash_object.hexdigest()
  return hex_dig


def align_paper_metadata(s2_papers_metadata: List[Dict]|Dict) -> List[Dict]:
    """
    Pre-processes Semantic Scholar paper metadata.
    - Use s2 paper id as primary id for paper
    - Extracts and normalizes ArXiv information.
    - Infers publication date if missing.
    - Limits the number of authors.

    Args:
        s2_papers_metadata: A single paper metadata dict or a list of dicts.

    Returns:
        A list of processed paper metadata dictionaries, each including a 'primary_id'.
    """
    if isinstance(s2_papers_metadata, dict):
        s2_papers_metadata = [s2_papers_metadata]

    processed_papers = []
    for item in s2_papers_metadata:
        if not isinstance(item, dict):
            # Handle cases where item is not a dictionary (e.g., log a warning)
            print(f"Warning: Skipping non-dictionary item in align_paper_metadata: {item}")
            continue

        s2_paper_id = item.get('paperId')
        if s2_paper_id is None:
            # Skip papers without a Semantic Scholar paperId
            print(f"Warning: Skipping paper due to missing 'paperId': {item.get('title', 'N/A')}")
            continue

        # --- Determine IDs (doi, arxiv_id, unique id) ---
        doi = item.get('externalIds',{}).get('DOI')  # doi
        arxiv_id_raw = item.get('externalIds',{}).get('ArXiv')  # arxiv id
        arxiv_id_rvsd = None # Canonical ArXiv ID (without version)

        # Handle cases where DOI is actually an ArXiv DOI
        if doi and doi.startswith('10.48550/arXiv.') and not arxiv_id_raw:
            arxiv_id_raw = doi.replace('10.48550/arXiv.', '') # Extract raw ArXiv ID from DOI

        if arxiv_id_raw:
            # Clean ArXiv ID: remove version, handle potential DOI prefix if missed
            arxiv_no_prefix = arxiv_id_raw.replace('10.48550/arXiv.', '')
            arxiv_id_rvsd = re.sub(r'v\d+$', '', arxiv_no_prefix) # Remove version
            version_match = re.search(r'v(\d+)$', arxiv_no_prefix) # Get version info (capture group)

            # Generate ArXiv related info
            item['version'] = f"v{version_match.group(1)}" if version_match else None # Store version like 'v1', 'v2'
            item['arxivUrl'] = f"https://arxiv.org/abs/{arxiv_no_prefix}"
            item['isOpenAccess'] = True # Assume ArXiv papers are OA
            item['openAccessPdf'] = {'url': f"https://arxiv.org/pdf/{arxiv_no_prefix}.pdf"} # Match S2 structure
            item['arxivId'] = arxiv_id_rvsd # Store canonical ArXiv ID
            
            # If DOI was missing, assign the canonical ArXiv DOI
            if not doi:
                doi = f"10.48550/arXiv.{arxiv_id_rvsd}" # Use canonical ArXiv ID for DOI

        # --- Assign Primary ID ---
        # Priority: DOI -> Canonical ArXiv ID -> S2 Paper ID
        item['id'] = s2_paper_id
        item['doi'] = doi # Keep original DOI field if it existed or was assigned

        # --- Process Publish Date ---
        publish_dt = item.get('publicationDate')
        year = item.get('year')
        if publish_dt is None:
            # Try inferring from ArXiv ID (basic inference for newer IDs)
            if arxiv_id_rvsd and re.match(r'^\d{4}\.', arxiv_id_rvsd): # Matches IDs like YYMM.xxxxx
                try:
                    parsed_year = int(f"20{arxiv_id_rvsd[:2]}") # Basic assumption for 20xx
                    parsed_month = int(arxiv_id_rvsd[2:4])
                    if 1 <= parsed_month <= 12:
                        item['publicationDate'] = f"{parsed_year:04d}-{parsed_month:02d}-01"
                    else:
                        raise ValueError("Invalid month")
                except (ValueError, IndexError):
                    # Fallback if parsing fails or format is unexpected
                    if year:
                        item['publicationDate'] = f"{year}-01-01"
                    else: # Optional: Add a very generic fallback?
                        item['publicationDate'] = '2000-01-01' # Or None
            elif year:
                item['publicationDate'] = f"{year}-01-01"
            else: # Optional fallback if year is also missing
                item['publicationDate'] = '2000-01-01' # Or None

        # --- Limit Authors ---
        authors_list = item.get('authors', [])
        if isinstance(authors_list, list):
            item['authors'] = authors_list[:10] # Limit to first 10 authors
        else:
            item['authors'] = [] # Set to empty list if 'authors' is not a list

        processed_papers.append(item)

    return processed_papers


def process_papers_data(
        s2_papers_metadata: List[Dict]|Dict,
        existing_nodes: Optional[Set[str]] = None,
        existing_edges: Optional[Set[Tuple[str, str, str]]] = None
        ) -> List[Dict]:
    """
    Transforms aligned paper metadata into Neo4j node and relationship JSON objects.
    Creates Paper, Author, Journal, Venue nodes and WRITES, PRINTS_ON, RELEASES_IN relationships.

    Args:
        aligned_papers_dict: List of processed paper metadata from align_paper_metadata.
        existing_nodes: Optional set to track node IDs already processed in a larger batch.
        existing_edges: Optional set to track edge tuples already processed in a larger batch.

    Returns:
        A list of Neo4j compatible node and relationship dictionaries.
        Example node: {'type': 'node', 'id': '...', 'labels': ['Paper'], 'properties': {...}}
        Example rel: {'type': 'relationship', 'relationshipType': 'WRITES', 'startNodeId': '...', 'endNodeId': '...', 'properties': {...}}
    """
    papers_json = []
    # Use provided sets or initialize new ones if running standalone
    _node_ids = existing_nodes if existing_nodes is not None else set()
    _edge_tuples = existing_edges if existing_edges is not None else set()

    # align ids and keys for paper metadata
    aligned_papers_dict = align_paper_metadata(s2_papers_metadata)
    
    # Define paper properties to include (excluding 'authors')
    paper_keys = ['paperId', 'doi', 'arxivId', 'title', 'abstract', 'year', 'publicationDate',
                  'citationCount', 'referenceCount', 'influentialCitationCount', 'arxivUrl',
                  'isOpenAccess', 'openAccessPdf', 'version',
                  'fieldsOfStudy'] 

    for item in aligned_papers_dict:
        paper_id = item.get('id')
        if paper_id is None:
            print(f"Warning: Skipping paper due to missing of primary id: {item.get('title', 'N/A')}")
            continue # Skip if no primary ID was assigned

        # --- Process Paper Node ---
        paper_props = {k: v for k, v in item.items() if k in paper_keys and v is not None}
        paper_node = {
            "type": "node",
            "id": paper_id,
            "labels": [NODE_PAPER],
            "properties": paper_props 
            }
        papers_json.append(paper_node)
        _node_ids.add(paper_id)

        # --- Process Authors ---
        authors = item.get('authors', [])[:10]
        for idx, author in enumerate(authors):
            if not isinstance(author, dict): continue # Skip if author is not a dict
            author_id = author.get('authorId')
            if author_id:
                if author_id not in _node_ids:
                    author_props = {k: v for k, v in author.items() if k in ['authorId', 'name'] and v is not None}
                    author_node = {
                        "type": "node",
                        "id": author_id,
                        "labels": [NODE_AUTHOR],
                        "properties": author_props}
                    papers_json.append(author_node)
                    _node_ids.add(author_id)
            
                # process author -> WRITES -> paper
                edge_tuple = (author_id, paper_id, REL_WRITES)
                if edge_tuple not in _edge_tuples:
                    author_order = idx + 1
                    author_paper_relationship = {
                        "type": "relationship",
                        "relationshipType": REL_WRITES,
                        "startNodeId": author_id,
                        "endNodeId": paper_id,
                        "properties": {'authorOrder': author_order}
                        }
                    papers_json.append(author_paper_relationship)
                    _edge_tuples.add(edge_tuple)

        # --- Process Journal ---
        journal = item.get('journal', {})
        journal_name = None
        paper_in_journal_props = {}

        if isinstance(journal, dict):
            journal_name = journal.get('name')  
            paper_in_journal_props['volume'] = journal.get('volume')
            paper_in_journal_props['pages'] = journal.get('pages')
            # Filter None values from properties
            paper_in_journal_props = {k:v for k,v in paper_in_journal_props.items() if v is not None}
        elif isinstance(journal, str):
            journal_name = journal # If journal is just a name string

        if journal_name and journal_name.strip() and 'arxiv' not in journal_name.lower():
            journal_hash_id = generate_hash_key(journal_name)
            if journal_hash_id and journal_hash_id not in _node_ids:
                journal_props = {"journal_hash_id": journal_hash_id, "name": journal_name, "hash_method":"hashlib.sha256"}
                journal_node = {
                    "type": "node",
                    "id": journal_hash_id,
                    "labels": [NODE_JOURNAL],
                    "properties": journal_props}
                papers_json.append(journal_node)
                _node_ids.add(journal_hash_id)
            
            # process paper -> PRINTS_ON -> journal
            edge_tuple = (paper_id, journal_hash_id, REL_PRINTS_ON)
            if edge_tuple not in _edge_tuples:
                if 'arxiv' not in journal_name.lower():  # exclude arxiv from journal
                    paper_journal_relationship = {
                        "type": "relationship",
                        "relationshipType": REL_PRINTS_ON,
                        "startNodeId": paper_id,
                        "endNodeId": journal_hash_id,
                        "properties": paper_in_journal_props}
                    papers_json.append(paper_journal_relationship)
                    _edge_tuples.add(edge_tuple)

        # --- Process Venue ---
        venue_info = item.get('publicationVenue') # S2: Can be string or object/dict
        venue_id = None
        venue_name = None
        venue_props_keys = ['id', 'name', 'type', 'url', 'alternate_names', 'issn'] # S2 Venue object fields
        venue_props = {}

        if isinstance(venue_info, dict):
            venue_id = venue_info.get('id') # Use S2 Venue ID if available
            venue_name = venue_info.get('name')
            venue_props = {k: v for k, v in venue_info.items() if k in venue_props_keys and v is not None}
        elif isinstance(venue_info, str):
            venue_name = venue_info # Venue is just a name string

        # Use hash of name if ID is missing but name exists
        if not venue_id and venue_name and venue_name.strip():
             venue_id = generate_hash_key(f"venue:{venue_name}") # Add prefix to avoid collision with other hashes
             if venue_id:
                 venue_props['venue_hash_id'] = venue_id
                 venue_props['name'] = venue_name
                 venue_props['hash_method'] = "hashlib.sha256"

        if venue_id and venue_name and 'arxiv' not in venue_name.lower():
            # Process Venue Node
            if venue_id not in _node_ids:
                # Ensure essential props are present if created via hash
                if 'id' not in venue_props: venue_props['id'] = venue_id
                if 'name' not in venue_props: venue_props['name'] = venue_name

                venue_node = {
                    "type": "node",
                    "id": venue_id,
                    "labels": [NODE_VENUE],
                    "properties": venue_props
                }
                papers_json.append(venue_node)
                _node_ids.add(venue_id)

            # Process RELEASES_IN Relationship
            edge_tuple = (paper_id, venue_id, REL_RELEASES_IN)
            if edge_tuple not in _edge_tuples:
                paper_venue_relationship = {
                    "type": "relationship",
                    "relationshipType": REL_RELEASES_IN,
                    "startNodeId": paper_id,
                    "endNodeId": venue_id,
                    "properties": {} # Add properties if relevant (e.g., publication type if available)
                }
                papers_json.append(paper_venue_relationship)
                _edge_tuples.add(edge_tuple)

    return papers_json


def process_authors_data(
        s2_authors: List[Dict]|Dict,
        existing_nodes: Optional[Set[str]] = None,
        existing_edges: Optional[Set[Tuple[str, str, str]]] = None
        ) -> List[Dict]:
    """
    Transforms Semantic Scholar author metadata into Neo4j node and relationship JSON objects.
    Creates Author, Institution nodes and WORKS_IN relationships.

    Args:
        s2_authors: A single author metadata dict or a list of dicts.
        existing_nodes: Optional set to track node IDs already processed in a larger batch.
        existing_edges: Optional set to track edge tuples already processed in a larger batch.

    Returns:
        A list of Neo4j compatible node and relationship dictionaries.
    """
    if isinstance(s2_authors, dict):
        s2_authors = [s2_authors]

    authors_json = []
    _node_ids = existing_nodes if existing_nodes is not None else set()
    _edge_tuples = existing_edges if existing_edges is not None else set()

    # --- Define properties to keep for Author nodes ---
    author_props_to_keep = [
        'authorId', 'externalIds', 'name', 'aliases', 'url', 'hIndex',
        'paperCount', 'citationCount', 'influentialCitationCount', 'homepage', 'semanticScholarUrl'
    ]

    for author_item in s2_authors:
        if not isinstance(author_item, dict):
            print(f"Warning: Skipping non-dictionary item in process_authors_data: {author_item}")
            continue

        author_id = author_item.get('authorId')
        if author_id is None:
            print(f"Warning: Skipping author due to missing 'authorId': {author_item.get('name', 'N/A')}")
            continue

        # --- Process Author Node ---
        author_props = {
            k: v for k, v in author_item.items()
            if k in author_props_to_keep and v is not None
        }
        # Ensure essential props are present
        if 'authorId' not in author_props: author_props['authorId'] = author_id
        if 'name' not in author_props: author_props['name'] = author_item.get('name', 'Unknown Name')

        author_node = {
            "type": "node",
            "id": author_id,
            "labels": [NODE_AUTHOR],
            "properties": author_props
        }
        authors_json.append(author_node)
        _node_ids.add(author_id)

        # --- Process Affiliations (Institutions and WORKS_IN) ---
        institutions = author_item.get('affiliations', []) # S2: list of strings or objects
        if isinstance(institutions, list):
            for inst_data in institutions:
                inst_name = None
                inst_id = None # S2 might provide an ID for institutions
                inst_props = {}

                if isinstance(inst_data, str):
                    inst_name = inst_data.strip()
                elif isinstance(inst_data, dict):
                    # Try to get name and ID from the institution object
                    inst_name = inst_data.get('name', inst_data.get('institution')) # Common keys
                    inst_id = inst_data.get('id') # Check if S2 provides an ID
                    # Capture other potentially useful props
                    inst_props = {k: v for k, v in inst_data.items() if k in ['name', 'id', 'type'] and v is not None}


                if inst_name and inst_name.strip():
                    # Determine Institution Node ID: Prefer S2 ID, fallback to hash of name
                    node_inst_id = inst_id
                    if not node_inst_id:
                        node_inst_id = generate_hash_key(f"institution:{inst_name}") # Add prefix
                        if node_inst_id:
                           inst_props['institution_hash_id'] = node_inst_id
                           inst_props['hash_method'] = 'hashlib.sha256'

                    if node_inst_id: # Proceed if we have a valid ID
                        # Process Institution Node
                        if node_inst_id not in _node_ids:
                            # Ensure essential props are present
                            if 'id' not in inst_props: inst_props['id'] = node_inst_id
                            if 'name' not in inst_props: inst_props['name'] = inst_name

                            inst_node = {
                                "type": "node",
                                "id": node_inst_id,
                                "labels": [NODE_INSTITUTION],
                                "properties": inst_props
                            }
                            authors_json.append(inst_node)
                            _node_ids.add(node_inst_id)

                        # Process WORKS_IN Relationship
                        edge_tuple = (author_id, node_inst_id, REL_WORKS_IN)
                        if edge_tuple not in _edge_tuples:
                            # Add relationship properties if available (e.g., start/end year)
                            rel_props = {} # Extract from author_item if S2 provides timeframe for affiliation
                            author_inst_relationship = {
                                "type": "relationship",
                                "relationshipType": REL_WORKS_IN,
                                "startNodeId": author_id,
                                "endNodeId": node_inst_id,
                                "properties": rel_props
                            }
                            authors_json.append(author_inst_relationship)
                            _edge_tuples.add(edge_tuple)

    return authors_json


def process_citations_data(
        s2_citations: Union[List[Dict], Dict],
        existing_edges: Optional[Set[Tuple[str, str, str]]] = None
        ) -> List[Dict]:
    """
    Transforms citation data into Neo4j CITES relationship JSON objects.
    Assumes input contains 'source_id' (citing paper primary_id) and
    'target_id' (cited paper primary_id).

    Args:
        s2_citations: A single citation dict or a list of dicts.
                      Each dict must contain 'source_id' and 'target_id'
                      matching the primary_ids used for paper nodes.
        existing_edges: Optional set to track edge tuples already processed in a larger batch.

    Returns:
        A list of Neo4j compatible relationship dictionaries.
    """
    if isinstance(s2_citations, dict):
        s2_citations = [s2_citations]

    citations_json = []
    _edge_tuples = existing_edges if existing_edges is not None else set()

    # Keys for citation properties within the citation record item
    citation_props_keys = ['citation_type', 'isInfluential', 'contexts', 'intents', 'contextsWithIntent']

    for item in s2_citations:
        if not isinstance(item, dict):
             print(f"Warning: Skipping non-dictionary item in process_citations_data: {item}")
             continue

        start_node_id = item.get('source_id') # ID of the citing paper
        end_node_id = item.get('target_id')   # ID of the cited paper

        if not start_node_id or not end_node_id:
            print(f"Warning: Skipping citation due to missing source/target ID: {item}")
            continue

        citation_props = {k: v for k, v in item.items()
                          if k in citation_props_keys and v is not None}

        # Process CITES Relationship
        edge_tuple = (start_node_id, end_node_id, REL_CITES)
        if edge_tuple not in _edge_tuples:
            paper_cites_relationship = {
                "type": "relationship",
                "relationshipType": REL_CITES,
                "startNodeId": start_node_id,
                "endNodeId": end_node_id,
                "properties": citation_props
            }
            citations_json.append(paper_cites_relationship)
            _edge_tuples.add(edge_tuple)

    return citations_json
    

def process_topics_data(
        topics_data: List[dict],
        existing_nodes: Optional[Set[str]] = None,
        existing_edges: Optional[Set[Tuple[str, str, str]]] = None
        ) -> List[Dict]:
    """
    Transforms topic data into Neo4j Topic nodes and DISCUSS relationship JSON objects.
    Assumes input contains 'topic' name and 'primary_id' (the paper's primary_id).

    Args:
        topics_data: A single topic dict or a list of dicts.
                     Each dict must contain 'topic' and 'id' (s2 paper ID).
        existing_nodes: Optional set to track node IDs already processed in a larger batch.
        existing_edges: Optional set to track edge tuples already processed in a larger batch.

    Returns:
        A list of Neo4j compatible node and relationship dictionaries.
    """
    if isinstance(topics_data, dict):
        topics_data = [topics_data]

    topics_json = []
    _node_ids = existing_nodes if existing_nodes is not None else set()
    _edge_tuples = existing_edges if existing_edges is not None else set()

    for item in topics_data:
        if not isinstance(item, dict):
            print(f"Warning: Skipping non-dictionary item in process_topics_data: {item}")
            continue

        topic_name = item.get('topic')
        # IMPORTANT: Use the same primary ID as generated for the paper node
        paper_id = item.get('paperId') # Changed from 's2id'

        if topic_name and topic_name.strip() and paper_id:
            topic_hash_id = generate_hash_key(f"topic:{topic_name}") # Add prefix
            if topic_hash_id: # Ensure hash was generated
                # Process Topic Node
                if topic_hash_id not in _node_ids:
                    topic_props = {
                        'topic_hash_id': topic_hash_id,
                        'name': topic_name, 
                        'hash_method': 'hashlib.sha256'
                        }
                    topic_node = {
                        'type': 'node',
                        'id': topic_hash_id,
                        'labels': [NODE_TOPIC],
                        'properties': topic_props
                    }
                    topics_json.append(topic_node)
                    _node_ids.add(topic_hash_id)

                # Process DISCUSS Relationship
                edge_tuple = (paper_id, topic_hash_id, REL_DISCUSS)
                if edge_tuple not in _edge_tuples:
                    # Add relationship properties if available (e.g., relevance score)
                    rel_props = {k:v for k,v in item.items() if k not in ['topic', 'id']}

                    paper_topic_relationship = {
                        "type": "relationship",
                        "relationshipType": REL_DISCUSS,
                        "startNodeId": paper_id,
                        "endNodeId": topic_hash_id,
                        "properties": rel_props
                    }
                    topics_json.append(paper_topic_relationship)
                    _edge_tuples.add(edge_tuple)
        else:
             print(f"Warning: Skipping topic item due to missing topic name or paper id: {item}")


    return topics_json