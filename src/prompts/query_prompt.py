query_example = [
  {
    "query": "social media political polarization youth influence",
    "description": "This query targets the core research question, focusing specifically on the impact or influence of social media platforms on the political polarization levels observed among young adults or adolescents. Useful for finding studies measuring both variables in this specific demographic."
  },
  {
    "query": "echo chamber filter bubble political polarization online exposure",
    "description": "Focusing on the mechanism suggested in the findings (ideologically homogeneous content), this query seeks studies investigating how exposure patterns online, such as echo chambers or filter bubbles, contribute to political polarization across various populations."
  },
  {
    "query": "mixed methods political communication media effects survey interview",
    "description": "This query emphasizes the methodology employed. It aims to find research within political communication or media effects that utilizes a mixed-methods approach, potentially combining quantitative (survey) and qualitative (interview) data, to study phenomena like polarization or attitude formation. Useful for methodological comparisons."
  },
  {
    "query": "digital media youth civic engagement political attitudes",
    "description": "This query explores related research topics by broadening the scope. It searches for literature examining the relationship between young people's use of digital media and their broader civic engagement levels or general political attitudes, which may provide context for polarization studies."
  }
]

query_prompt = """## TASK
You are a sophisticated academic scholar skillful with search engines. 
Given the following paper contexts, you are asked to compose 3-5 search queries help find more relevant literature works.

## INSTRUCTION
- Generate queries with different emphasis on research questions, methodologies, inherited or related research topics, etc. 
- Design queries as keyword-focused, they would be used to find matches in other papers' title, abstract and context. Restrict to use operators like "OR", "AND" as few as possible.
- Make each query shall distinct and non-overlap with each other. They lead to broader, more diversified yet related papers through search results.
- Give descriptions and extra information on the query, which could help further filter the search results.

Please output in json format the example shows. Only one set of queries for all the papers from input.

## EXAMPLE

Here is an example:
```
<paper> Social media usage heighten political polarization in youth - A quantitative study 
This study employed a mixed-methods approach to investigate the impact of social media usage on political polarization among young adults in urban areas. 
Quantitative data was collected through a survey of 500 participants, while qualitative data was gathered via semi-structured interviews with a subset of 25 participants. 
The findings suggest a correlation between increased exposure to ideologically homogeneous content online and heightened political polarization.
</paper>
```

Hypothetical output for this example:
```json
{query_example}
```

## INPUT
Now start analyzing the following texts from paper(s).

```
{input_text}
```

## OUTPUT
Please think step-by-step and finally give your answer here:

"""