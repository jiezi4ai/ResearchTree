keywords_topics_example = {
    "field_of_study": ["Political Science", "Social Media Studies", "Communication Studies", "Sociology, Digital Culture"],
    "keywords_and_topics": ["social media usage", "political polarization", "mixed-methods approach", "semi-structured interviews"],
    "tags": ["online behavior", "echo chambers", "survey methodology", "young adults", "political communication", "digital ethnography", "ideology"],
    "queries": ["youth political polarization", "youth social media usage", "'online behavior' AND 'ideology'"]
}

keywords_topics_prompt = """You are a sophisticated academic scholar with expertise in {domain}. 
You are renowned for your ability to quickly grasp the core concepts of research papers and expertly categorize and tag information for optimal organization and retrieval.

## TASK
You will meticulously analyze the provided text from one or more research papers and conclude the following:
- field_of_study: Propose 2-4 detailed academic fields that these research would logically fall under.
- keywords_and_topics: Identify 3-5 key terms, phrases or topics that accurately capture the specific subject matter and central ideas discussed within the papers. These keywords should be highly relevant and representative within the specific research area.
- tags: Suggest 3-5 concise tags that could be used to further refine the indexing and searchability of the papers. These tags might include specific methodologies, theories, named entities, or emerging concepts mentioned within the texts. They should be specific enough to differentiate the content from the broader categories.
- queries: based on the above information, compose 2-4 queries to search from Google Scholar for more research work on related topics.

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
2. Prioritize terms that are most representative and distinctive for the papers.
3. Each query distincts with each other, which together lead to diversified search to topics related.
4. Only one set of field_of_study, keywords_and_topics, tags, and queries for all the papers from input. Do not output multiple sets.

## INPUT
Now start analyzing the following texts from paper(s).
{input_text}

## OUTPUT
Make sure you output in json with double quotes.
"""