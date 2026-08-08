"""
Phase 6a — Index the item catalog into Elasticsearch.

Two kinds of fields go into each document, and they are NOT the same kind
of data — this distinction matters and should be represented honestly if
this project comes up in an interview:

  REAL fields (from item_features, computed from actual user behavior):
    item_id, category_id, total_events, buy_count, view_count,
    distinct_users, popularity_score
  These support genuine structured/faceted search: filter by category,
  sort by popularity or purchase count, etc. — fully legitimate, no
  caveats needed.

  SYNTHETIC field (title):
    The Taobao UserBehavior dataset has NO real product text — no titles,
    no descriptions, only numeric item_id/category_id. Real full-text
    search needs text to search over, so this script generates a
    placeholder title like "Item 12345 (Category 2885642)" purely so
    Elasticsearch's actual full-text matching/relevance-scoring behavior
    can be demonstrated. This is clearly synthetic, not real product data
    — describe it that way if asked, rather than implying real product
    names were searched.

Usage:
    python src/pipeline/12_index_elasticsearch.py \
        --features data/processed \
        --es-host http://localhost:9200 \
        --index items
"""

import argparse

import pandas as pd
from elasticsearch import Elasticsearch, helpers


INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "item_id": {"type": "long"},
            "category_id": {"type": "long"},
            "title": {"type": "text"},          # synthetic — see module docstring
            "total_events": {"type": "integer"},
            "buy_count": {"type": "integer"},
            "view_count": {"type": "integer"},
            "distinct_users": {"type": "integer"},
            "popularity_score": {"type": "float"},
        }
    }
}


def make_title(item_id: int, category_id) -> str:
    """Synthetic placeholder text — see module docstring. Real product
    search would use an actual product title/description here instead."""
    cat = int(category_id) if pd.notna(category_id) else "unknown"
    return f"Item {item_id} (Category {cat})"


def generate_docs(df: pd.DataFrame, index: str):
    for row in df.itertuples(index=False):
        doc = {
            "item_id": int(row.item_id),
            "category_id": int(row.category_id) if pd.notna(row.category_id) else None,
            "title": make_title(row.item_id, row.category_id),
            "total_events": int(row.total_events),
            "buy_count": int(row.buy_count),
            "view_count": int(row.view_count),
            "distinct_users": int(row.distinct_users),
            "popularity_score": float(row.popularity_score),
        }
        yield {"_index": index, "_id": doc["item_id"], "_source": doc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="dir with item_features parquet")
    parser.add_argument("--es-host", default="http://localhost:9200")
    parser.add_argument("--index", default="items")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate the index first")
    args = parser.parse_args()

    es = Elasticsearch(args.es_host)
    if not es.ping():
        raise SystemExit(f"Could not connect to Elasticsearch at {args.es_host}")
    print(f"Connected to Elasticsearch at {args.es_host}")

    if args.recreate and es.indices.exists(index=args.index):
        es.indices.delete(index=args.index)
        print(f"Deleted existing index '{args.index}'")

    if not es.indices.exists(index=args.index):
        es.indices.create(index=args.index, body=INDEX_MAPPING)
        print(f"Created index '{args.index}' with mapping")

    print(f"Loading {args.features}/item_features ...")
    df = pd.read_parquet(f"{args.features}/item_features")
    df = df.fillna(0)
    print(f"  {len(df):,} items to index")

    print("Indexing (this streams in bulk batches) ...")
    success, errors = helpers.bulk(es, generate_docs(df, args.index), stats_only=False, raise_on_error=False)
    print(f"Done. Indexed {success:,} documents.")
    if errors:
        print(f"  WARNING: {len(errors)} documents failed to index. First error:")
        print(f"    {errors[0]}")

    es.indices.refresh(index=args.index)
    count = es.count(index=args.index)["count"]
    print(f"Index '{args.index}' now contains {count:,} documents.")


if __name__ == "__main__":
    main()
