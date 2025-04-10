import asyncio
import json

import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from collect.paper_query import PaperQuery

async def main():
    research_topic = "llm literature review"
    seed_dois = ['10.48550/arXiv.2406.10252',  # AutoSurvey: Large Language Models Can Automatically Write Surveys
                '10.48550/arXiv.2412.10415',  # Generative Adversarial Reviews: When LLMs Become the Critic
                '10.48550/arXiv.2402.12928',  # A Literature Review of Literature Reviews in Pattern Analysis and Machine Intelligence 
                ]
    seed_titles = ['PaperRobot: Incremental Draft Generation of Scientific Ideas',
                'From Hypothesis to Publication: A Comprehensive Survey of AI-Driven Research Support Systems'
                ]
    
    pq = PaperQuery()

    processed_results = await pq.get_paper_info(
        research_topic=research_topic, 
        seed_paper_titles=seed_titles,
        seed_paper_dois=seed_dois,
        limit=50, 
        from_dt='2020-01-01', 
        to_dt='2025-04-30')
    
    with open("paper_query_test.json", "w") as f:
        json.dump(processed_results, f, indent=2)
    return processed_results

if __name__ == "__main__":
    asyncio.run(main())