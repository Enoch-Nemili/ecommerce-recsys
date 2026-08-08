"""
Phase 6b — Search demo: structured filters + full-text search.

Run this after 12_index_elasticsearch.py to see Elasticsearch actually
working against your real catalog. Demonstrates three genuinely different
query types Elasticsearch is good at:

  1. Structured filter + sort (100% real data): "most popular items in a
     given category" — a term filter plus a sort, no text involved.
  2. Full-text search (against the synthetic title field): fuzzy/relevance
     matching — the kind of search a real product-title field would use.
  3. A combined query: text match AND a structured filter together,
     boosted by a real signal (popularity) — this is closer to how a real
     product search page actually ranks results, blending relevance with
     business signals rather than using either alone.

Usage:
    python src/pipeline/13_search_demo.py --es-host http://localhost:9200 --index items
"""

import argparse
import json

from elasticsearch import Elasticsearch


def run_query(es, index, name, body, size=5):
    print(f"\n=== {name} ===")
    body = dict(body)
    body["size"] = size
    result = es.search(index=index, body=body)
    hits = result["hits"]["hits"]
    print(f"  {result['hits']['total']['value']:,} total matches, showing top {len(hits)}:")
    for h in hits:
        src = h["_source"]
        # Elasticsearch returns _score: null whenever results are sorted by
        # a custom field (like popularity_score) rather than by relevance —
        # that's expected, not an error, so it needs its own display case
        # rather than assuming a number is always there to format.
        score_str = f"{h['_score']:.3f}" if h["_score"] is not None else "N/A (sorted, not relevance-ranked)"
        print(f"    item {src['item_id']:>10}  score={score_str}  "
              f"popularity={src['popularity_score']:.1f}  category={src['category_id']}  "
              f"title=\"{src['title']}\"")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--es-host", default="http://localhost:9200")
    parser.add_argument("--index", default="items")
    parser.add_argument("--category", type=int, default=None,
                         help="A real category_id from your data, for the filter demo")
    args = parser.parse_args()

    es = Elasticsearch(args.es_host)
    if not es.ping():
        raise SystemExit(f"Could not connect to Elasticsearch at {args.es_host}")

    # 1. Structured: most popular items overall (real data, no text involved)
    run_query(es, args.index, "Top items by popularity_score (structured sort)", {
        "query": {"match_all": {}},
        "sort": [{"popularity_score": {"order": "desc"}}],
    })

    # 2. Structured filter, if a category was given
    if args.category is not None:
        run_query(es, args.index, f"Items in category {args.category}, sorted by popularity", {
            "query": {"term": {"category_id": args.category}},
            "sort": [{"popularity_score": {"order": "desc"}}],
        })

    # 3. Full-text search against the synthetic title field
    run_query(es, args.index, "Full-text search: 'Item 123' (fuzzy match demo)", {
        "query": {
            "match": {
                "title": {"query": "Item 123", "fuzziness": "AUTO"},
            }
        },
    })

    # 4. Combined: text relevance + a real business signal (popularity) boosting the score
    run_query(es, args.index, "Combined: text relevance + popularity boost", {
        "query": {
            "function_score": {
                "query": {"match": {"title": "Item"}},
                "functions": [
                    {"field_value_factor": {"field": "popularity_score", "modifier": "log1p", "missing": 0}}
                ],
                "boost_mode": "sum",
            }
        },
    })


if __name__ == "__main__":
    main()
