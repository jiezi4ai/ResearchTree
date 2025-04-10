import asyncio
import json

import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from collect.citation_query import CitationQuery


async def main():
    cq = CitationQuery(
        from_dt = '2024-01-01',
        to_dt = '2025-04-30',
        fields_of_study = ['Computer Science'],
    )
    paper_doi = '10.48550/arXiv.2406.10252'
    cited_papers_info = await cq.get_cited_papers(paper_doi=paper_doi)
    print(f"Get {len(cited_papers_info)} cited paper information!")

    with open("citation_query_test_cited.json", "w") as f:
        json.dump(cited_papers_info, f, indent=2)
    
    citing_papers_info = await cq.get_citing_papers(paper_doi=paper_doi)
    print(f"Get {len(citing_papers_info)} citing paper information!")

    with open("citation_query_test_citing.json", "w") as f:
        json.dump(citing_papers_info, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())