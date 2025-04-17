import os
import json
import asyncio

import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from paper_collect import PaperCollector

async def main():
    llm_api_key = os.getenv('GEMINI_API_KEY_3')
    llm_model_name="gemini-2.0-flash"
    embed_api_key = os.getenv('GEMINI_API_KEY_3')
    embed_model_name="models/text-embedding-004"

    research_topic = "llm literature review"
    seed_dois = ['10.48550/arXiv.2406.10252',  # AutoSurvey: Large Language Models Can Automatically Write Surveys
                '10.48550/arXiv.2412.10415',  # Generative Adversarial Reviews: When LLMs Become the Critic
                '10.48550/arXiv.2402.12928',  # A Literature Review of Literature Reviews in Pattern Analysis and Machine Intelligence 
                ]
    seed_titles = ['PaperRobot: Incremental Draft Generation of Scientific Ideas',
                'From Hypothesis to Publication: A Comprehensive Survey of AI-Driven Research Support Systems'
                ]


    pc = PaperCollector(   
        # research_topic = research_topic,   
        # seed_paper_titles = seed_titles, 
        seed_paper_dois = seed_dois[0],
        llm_api_key = llm_api_key,
        llm_model_name = llm_model_name,
        embed_api_key = embed_api_key,
        embed_model_name = embed_model_name,
        from_dt = '2020-01-01',
        to_dt = '2025-04-30',
        fields_of_study = ['Computer Science'])


    await pc.construct_paper_graph(
        search_citation = None,  # 'both',
        search_author = True,
        find_recommend = True,
        if_related_topic = False,
        if_expanded_citations  = None,  #  'reference',
        if_expanded_authors = False,
        if_add_similarity = False,
        similarity_threshold = 0.7,
        expanded_k_papers = 10,
        expanded_l_authors = 50,
    )

    with open("paper_collect_test_nodes.json", "w") as f:
        json.dump(pc.nodes_json, f, indent=2)

    with open("paper_collect_test_edges.json", "w") as f:
        json.dump(pc.edges_json, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())