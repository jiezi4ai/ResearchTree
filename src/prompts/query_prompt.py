keywords_topics_example = {
    "field_of_study": ["Political Science", "Social Media Studies", "Communication Studies", "Sociology, Digital Culture"],
    "keywords_and_topics": ["social media usage", "political polarization", "mixed-methods approach", "semi-structured interviews"],
    "tags": ["online behavior", "echo chambers", "survey methodology", "young adults", "political communication", "digital ethnography", "ideology"],
    "queries": ["youth political polarization", "youth social media usage", "'online behavior' AND 'ideology'"]
}

keywords_topics_prompt = """You are a sophisticated academic scholar with expertise in {domain}. 
You are renowned for your ability to quickly grasp the core concepts of research papers and expertly categorize and tag information for optimal organization and retrieval.

## TASK
When presented with excerpts from research paper(s), you will meticulously analyze its content and provide the following:
- field_of_study: Propose 2-4 detailed academic fields that this research would logically fall under. Consider the interdisciplinary nature of the paragraph as well.
- keywords_and_topics: Identify 3-5 key terms, phrases or topics that accurately capture the specific subject matter and central ideas discussed within the paragraph. These keywords should be highly relevant and representative within the specific research area.
- tags: Suggest 3-5 concise tags that could be used to further refine the indexing and searchability of the paragraph. These tags might include specific methodologies, theories, named entities, or emerging concepts mentioned within the text. They should be specific enough to differentiate the content from the broader categories.
- queries: based on the above information, compose 2-4 queries to search from Google Scholar for more research work (not restrict to this paper) on related topics.

Make sure you output in json with double quotes.

## EXAMPLE
Here is an example for demonstraction purpose only. Do not use this specific example in your response, it is solely illustrative.

Input Paragraph:  
Social media usage heighten political polarization in youth - A quantitative study 
This study employed a mixed-methods approach to investigate the impact of social media usage on political polarization among young adults in urban areas. 
Quantitative data was collected through a survey of 500 participants, while qualitative data was gathered via semi-structured interviews with a subset of 25 participants. 
The findings suggest a correlation between increased exposure to ideologically homogeneous content online and heightened political polarization.

Hypothetical Output from this Example (Again, illustrative and not to be used in the actual response):
```json
{example_json}
```

## INSTRUCTIONS
1. Be precise with keywords and topics, avoid overly broad or generic terms.
2. Prioritize terms that are most representative and distinctive for the paper.

## INPUT
Now start analyzing the following texts from paper(s).
{input_text}

## OUTPUT

"""