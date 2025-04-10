import asyncio

import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)


from src.apis.s2_api import SemanticScholarKit


async def test(): 
    s2 = SemanticScholarKit()
    test_papers = await s2.async_search_paper_by_ids(id_list=["10.48550/arXiv.2412.10415"])

    searches = await s2.async_search_paper_by_keywords(query="llm long term memory")

if __name__ == "__main__":
    asyncio.run(test())
