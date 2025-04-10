import asyncio
import json

import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from collect.paper_recommendation import PaperRecommendation

async def main():
    seed_dois = ['10.48550/arXiv.2406.10252',  # AutoSurvey: Large Language Models Can Automatically Write Surveys
                '10.48550/arXiv.2412.10415',  # Generative Adversarial Reviews: When LLMs Become the Critic
                '10.48550/arXiv.2402.12928',  # A Literature Review of Literature Reviews in Pattern Analysis and Machine Intelligence 
                ]
    
    pq = PaperRecommendation(
        from_dt = '2024-01-01',
        to_dt = '2025-04-30',
        fields_of_study = ['Computer Science'],
    )

    recommendations = await pq.get_recommend_papers(paper_dois=seed_dois)
    
    with open("paper_recommendations_test.json", "w") as f:
        json.dump(recommendations, f, indent=2)
    return recommendations

if __name__ == "__main__":
    asyncio.run(main())