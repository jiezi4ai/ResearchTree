import json
import asyncio
from typing import List, Dict, Optional, Union, Tuple

import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from collect.paper_query import PaperQuery
from collect.related_topic_query import RelatedTopicQuery

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def search(seed_dois):
    pq = PaperQuery()

    processed_results = await pq.get_paper_info(
        seed_paper_dois=seed_dois,
        limit=50, 
        from_dt='2020-01-01', 
        to_dt='2025-04-30')
    
    return processed_results


async def expand(
        seed_paper_json: Optional[List[Dict]],
        llm_api_key: Optional[str],
        llm_model_name: Optional[str],
        search_limit: Optional[int] = 50,
        from_dt='2022-01-01',
        to_dt='2025-03-13',
        fields_of_study=None,
):
    # --- Task Prep: Expanded Search ---
    # Needs to run sequentially internally (LLM -> Search) but concurrently with others
    expanded_search_task = None

    tq = RelatedTopicQuery(llm_api_key=llm_api_key, llm_model_name=llm_model_name)
    async def _handle_expanded_search():
        logging.info("Starting expanded search sub-task...")
        keywords_topics_json = await tq.llm_gen_related_topics(seed_paper_json)
        print(keywords_topics_json)

        seed_paper_topic_json = tq.get_topic_json(keywords_topics_json, seed_paper_json)

        related_papers_items = []
        if keywords_topics_json:
            # Now fetch related papers based on the generated topics
            try:
                related_papers_items = await tq.get_related_papers(keywords_topics_json, search_limit, from_dt, to_dt, fields_of_study)
            except Exception as e:
                logging.error(f"Failed to get related topicL {e}.")
        else:
            logging.warning("Skipping related paper fetch as no topics were generated.")

        expaned_items_json = seed_paper_topic_json + related_papers_items
        return expaned_items_json

    logging.info("Preparing expanded search task.")
    expanded_search_task = asyncio.create_task(_handle_expanded_search()) # Create task immediately

    logging.info("Waiting for expanded search task to complete...")
    try:
        expanded_result = await expanded_search_task
        logging.info("Expanded search task finished.")
        return expanded_result
    except Exception as e:
        logging.error(f"Expanded search task failed: {e}")
        return []


async def main():
    seed_dois = ['10.48550/arXiv.2406.10252',  # AutoSurvey: Large Language Models Can Automatically Write Surveys
                '10.48550/arXiv.2412.10415',  # Generative Adversarial Reviews: When LLMs Become the Critic
                '10.48550/arXiv.2402.12928',  # A Literature Review of Literature Reviews in Pattern Analysis and Machine Intelligence 
                ]
    api_key = os.getenv('GEMINI_API_KEY_1')
    model_name = 'gemini-2.0-flash-exp'
    
    processed_results = await search(seed_dois)

    seed_paper_json = [node for node in processed_results 
                       if node.get('labels') == ['Paper'] and node['properties'].get('from_seed')==True]
    
    expaned_items_json = await expand(seed_paper_json=seed_paper_json, llm_api_key=api_key, llm_model_name=model_name)

    with open("related_topic_query_test.json", "w") as f:
        json.dump(expaned_items_json, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())