"""
Phase 4b — Build a FAISS index over item embeddings.

Takes the item embeddings from Phase 4a and builds a FAISS index — a data
structure that can answer "which items are closest to this vector" in
milliseconds, even across millions of items, instead of comparing against
every item one by one.

Also provides a query function to test retrieval: given a user_id, look up
their embedding and find the top-K nearest items — this is a second,
embedding-based candidate source that complements the co-visitation
candidates from Phase 1 (Phase 1's candidates only know about items that
have literally co-occurred together; embeddings can generalize to items
that never directly co-occurred but are conceptually similar).

Usage:
    python src/pipeline/09_build_faiss_index.py \
        --embeddings data/processed \
        --output data/processed \
        --query-user <some_real_user_id>   # optional sanity check
"""

import argparse
import json

import faiss
import numpy as np


def build_index(item_embeddings: np.ndarray) -> faiss.Index:
    """Uses inner product (dot product) as the similarity metric, matching
    how the two-tower model was trained (score = dot product of user and
    item vectors) — using a different metric here (e.g. L2 distance) would
    silently give inconsistent results with what the model actually
    learned."""
    dim = item_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(item_embeddings.astype(np.float32))
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, help="dir with item_embeddings.npy etc.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--query-user", type=int, default=None, help="Optional: a real user_id to test retrieval with")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    print("Loading embeddings ...")
    item_embeddings = np.load(f"{args.embeddings}/item_embeddings.npy")
    user_embeddings = np.load(f"{args.embeddings}/user_embeddings.npy")
    with open(f"{args.embeddings}/embedding_id_maps.json") as f:
        maps = json.load(f)
    item_id_to_idx = maps["item_id_to_idx"]
    idx_to_item_id = {v: k for k, v in item_id_to_idx.items()}
    user_id_to_idx = maps["user_id_to_idx"]

    print(f"Building FAISS index over {item_embeddings.shape[0]:,} items (dim={item_embeddings.shape[1]}) ...")
    index = build_index(item_embeddings)

    faiss.write_index(index, f"{args.output}/item_faiss.index")
    print(f"Done. Index written to {args.output}/item_faiss.index")

    if args.query_user is not None:
        uid_str = str(args.query_user)
        if uid_str not in user_id_to_idx:
            print(f"\nWARNING: user_id {args.query_user} not found in embedding_id_maps.json — skipping test query.")
            return

        u_idx = user_id_to_idx[uid_str]
        u_vec = user_embeddings[u_idx:u_idx + 1].astype(np.float32)

        scores, indices = index.search(u_vec, args.top_k)
        print(f"\nTop {args.top_k} embedding-based candidates for user {args.query_user}:")
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            item_id = idx_to_item_id[idx]
            print(f"  {rank}. item {item_id}  (score {score:.4f})")


if __name__ == "__main__":
    main()
