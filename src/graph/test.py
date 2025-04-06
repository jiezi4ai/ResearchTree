import json
import asyncio
from typing import List, Dict, Optional, Union, Tuple # Added Tuple

from graph.paper_collect import PaperSearch
from graph.paper_graph import PaperGraph
class PaperExploration:

    def __init__(
            self,
            rounds: int = 2,
            research_topic: Optional[str]=None,
            seed_paper_dois: Optional[List[str]]=None,
            seed_paper_titles: Optional[List[str]]=None,
            llm_api_key: Optional[str] = None,
            llm_model_name: Optional[str] = None,
            embed_api_key: Optional[str] = None,
            embed_model_name: Optional[str] = None,
            ): 
        # Initialize with a seed DOI
        self.searcher = PaperSearch(
            research_topic=research_topic,
            seed_paper_dois=seed_paper_dois,
            seed_paper_titles=seed_paper_titles,
            llm_api_key=llm_api_key,
            llm_model_name=llm_model_name,
            embed_api_key=embed_api_key,
            embed_model_name=embed_model_name,
        )

    def save_nodes_and_edges(self):
        # Optional: Save or inspect searcher.nodes_json and searcher.edges_json
        with open("nodes.json", "w") as f:
            json.dump(self.searcher.nodes_json, f, indent=2)
        with open("edges.json", "w") as f:
            json.dump(self.searcher.edges_json, f, indent=2)

    async def round_1_search(
            self,
            with_reference: Optional[bool]=True,
            with_author: Optional[bool]=True, # Fetch authors of seed paper
            with_recommend: Optional[bool]=True,
            with_expanded_search: Optional[bool]=True, # Set to True to test LLM topic generation
            add_semantic_similarity: Optional[bool]=True,
            similarity_threshold: Optional[bool]=0.7,
            search_limit: Optional[int] = 50,
            recommend_limit: Optional[int] = 50,
            citation_limit: Optional[int] = 100,
            from_dt: Optional[str]="2000-01-01",
            to_dt: Optional[str]="9999-12-31"
        ):
        print("--- Running Initial Query ---")
        await self.searcher.init_collect(limit=search_limit, from_dt="2015-01-01", to_dt="2025-04-01")
        print(f"Nodes after init: {len(self.searcher.nodes_json)}")
        print(f"Edges after init: {len(self.searcher.edges_json)}")

        # Get the actual seed DOIs found (in case input was title/topic)
        # For this example, we know the input DOI, but in general:
        seed_dois_in_graph = [node['id'] for node in self.searcher.nodes_json if node['labels'] == ['Paper'] and node['properties'].get('from_seed')]
        if not seed_dois_in_graph:
            print("Warning: Seed DOI(s) not found after initial query.")
            return

        print(f"\n--- Running Main Collection for seed DOIs: {seed_dois_in_graph} ---")
        start_time = asyncio.get_event_loop().time()
        await self.searcher.collect(
            seed_paper_dois=seed_dois_in_graph, 
            with_reference=with_reference,
            with_author=with_author, # Fetch authors of seed paper
            with_recommend=with_recommend,
            with_expanded_search=with_expanded_search, 
            add_semantic_similarity=add_semantic_similarity,
            similarity_threshold=similarity_threshold,
            search_limit=search_limit,
            recommend_limit=recommend_limit,
            citation_limit=citation_limit,
            from_dt=from_dt,
            to_dt=to_dt
        )
        end_time = asyncio.get_event_loop().time()
        print(f"\nOptimized collection took {end_time - start_time:.2f} seconds.")
        print(f"Final Nodes: {len(self.searcher.nodes_json)}")
        print(f"Final Edges: {len(self.searcher.edges_json)}")


    def import_to_graph(
            self, 
            nodes_json: Optional[List] = None, 
            edges_json: Optional[List] = None):
        nodes_json = nodes_json if nodes_json is not None else self.searcher.nodes_json
        edges_json = edges_json if edges_json is not None else self.searcher.edges_json
        pg = PaperGraph(name="paper_graph")
        pg.add_graph_nodes(nodes_json)
        pg.add_graph_edges(edges_json)


    def filter_seed_ids(
            self,
            sub_graph,
            init_nodes,
    ):
        candit_items = []
        for u, v, wt in sub_graph[0].edges.data('weight'):
            # weight = sub_graph[0][u][v].get('weight')
            if wt and ((u in init_nodes and wt > 0.7 and wt < 0.95) or 
            (v in init_nodes and wt > 0.7 and wt < 0.95)):
                candit_items.append((u, v, wt))
        sorted_items = sorted(candit_items, key=lambda item: item[2], reverse=True)
        # for reference
        # similarity high
        k = 10
        i = 0
        new_seed_ids = []
        for u, v, sim in sorted_items:
            if i < k:
                if u not in init_nodes:
                    opt_node_id = u
                else:
                    opt_node_id = v
                new_seed_ids.append(opt_node_id)
                i += 1
            else:
                break
        return new_seed_ids
    
    def expand_seed(self, seed_ids):
        await self.searcher.collect(
            seed_paper_dois=seed_ids, 
            with_reference=True,
            similarity_threshold=similarity_threshold,
            search_limit=search_limit,
            recommend_limit=recommend_limit,
            citation_limit=citation_limit,
            from_dt=from_dt,
            to_dt=to_dt
        )
        end_time = asyncio.get_event_loop().time()
        print(f"\nOptimized collection took {end_time - start_time:.2f} seconds.")
        print(f"Final Nodes: {len(self.searcher.nodes_json)}")
        print(f"Final Edges: {len(self.searcher.edges_json)}")    
        
    # Round 2. Run initial query to get seed paper info
    seed_dois_in_graph

    print("--- Running Initial Query ---")
    await searcher.init_collect(limit=50, from_dt="2015-01-01", to_dt="2025-04-01")
    print(f"Nodes after init: {len(searcher.nodes_json)}")
    print(f"Edges after init: {len(searcher.edges_json)}")

    # Get the actual seed DOIs found (in case input was title/topic)
    # For this example, we know the input DOI, but in general:
    seed_dois_in_graph = [node['id'] for node in searcher.nodes_json if node['labels'] == ['Paper'] and node['properties'].get('from_seed')]
    if not seed_dois_in_graph:
         print("Warning: Seed DOI(s) not found after initial query.")
         # Handle error or exit
         return

    print(f"\n--- Running Main Collection for seed DOIs: {seed_dois_in_graph} ---")
    start_time = asyncio.get_event_loop().time()
    await searcher.collect(
        seed_paper_dois=seed_dois_in_graph, 
        with_reference=True,
        with_author=True, # Fetch authors of seed paper
        with_recommend=True,
        with_expanded_search=True, # Set to True to test LLM topic generation
        add_semantic_similarity=True,
        similarity_threshold=0.6,
        limit=100, # Limit number of items per fetch step
        from_dt="2020-01-01",
        to_dt="2025-04-01"
    )
    end_time = asyncio.get_event_loop().time()
    print(f"\nOptimized collection took {end_time - start_time:.2f} seconds.")
    print(f"Final Nodes: {len(searcher.nodes_json)}")
    print(f"Final Edges: {len(searcher.edges_json)}")

