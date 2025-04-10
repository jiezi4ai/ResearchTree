import asyncio
import json

import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from collect.author_query import AuthorQuery


async def main():
    aq = AuthorQuery(
        from_dt = '2024-01-01',
        to_dt = '2025-04-30',
        fields_of_study = ['Computer Science'],
    )
    author_ids = ['2108024279', '2273779175', '2286328804', '2116271777', '2262020955']
    authors_info = await aq.get_author_info(author_ids)
    print(f"Get {len(authors_info)} authors information!")

    with open("author_query_test.json", "w") as f:
        json.dump(authors_info, f, indent=2)
    

if __name__ == "__main__":
    asyncio.run(main())