query_example = [
    {
        "query": "'social media' AND ('political polarization' OR 'ideological polarization') AND ('youth' OR 'young adults')",
        "description": "Focuses on quantitative studies examining the relationship between social media use and political polarization specifically among young people. Good for finding similar methodological approaches and comparative studies."
    },
    {
        "query": "'echo chambers' OR 'filter bubbles' OR 'selective exposure' AND 'political attitudes' AND 'social media'",
        "description": "Targets the mechanism of how social media might cause polarization through information filtering and exposure. Useful for theoretical framework and causal explanations."
    },
    {
        "query": "'digital media' AND 'political behavior' OR (polarization OR radicalization)",
        "description": "Broadens the scope to general digital media effects on political behavior while excluding platform-specific studies. Helpful for understanding the wider context of media influence on political attitudes."
    },
    {
        "query": "'political polarization' AND ('social media' OR 'online media' AND ('measurement' OR 'quantitative study') ",
        "description": "Focuses on methodological approaches to measuring political polarization in digital contexts. Valuable for finding advanced measurement techniques and methodological discussions."
    }
]

query_prompt = """## TASK
You are a sophisticated academic scholar skillful with search engines. 
Given the following paper contexts, you are asked to compose 3-5 search queries help find more relevant literature works.

## INSTRUCTION
- Each query shall distinct and non-overlap with each other. All queries together cover main themes in input. 
- The queries lead to broader, more diversified yet related papers through search results.
- The queries should be keyword-focused. Utilize operators like "OR", "AND", "-" to improve query if necessary.
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