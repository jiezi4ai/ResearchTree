import asyncio
from s2_api import SemanticScholarKit


async def test(): 
    s2 = SemanticScholarKit()
    test_papers = await s2.async_search_paper_by_ids(id_list=["10.48550/arXiv.2412.10415"])

    searches = await s2.async_search_paper_by_keywords(query="llm long term memory")

if __name__ == "__main__":
    # Make sure to replace API keys before running
    # Configure logging level if needed
    # logging.getLogger().setLevel(logging.DEBUG) # For more verbose output
    asyncio.run(test())
