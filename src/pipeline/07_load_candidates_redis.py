"""
Phase 3b — Load precomputed user candidate lists into Redis.

Stores each user's candidate items as a Redis Sorted Set (ZSET), scored by
their co-visitation score. A sorted set lets the Java service fetch a
user's candidates already ranked, with one command, in the right order —
no extra sorting needed at request time.

Key design:
    cands:{user_id}  -> ZSET of {item_id: score}

Usage:
    python src/pipeline/07_load_candidates_redis.py \
        --input data/processed/user_candidates \
        --redis-host localhost \
        --redis-port 6379
"""

import argparse
from collections import defaultdict

import pandas as pd
import redis
from tqdm import tqdm

BATCH_SIZE = 2000  # users per Redis pipeline flush


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to user_candidates parquet")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--flush", action="store_true", help="Clear existing cands:* keys before loading")
    args = parser.parse_args()

    r = redis.Redis(host=args.redis_host, port=args.redis_port, db=args.redis_db, decode_responses=True)
    r.ping()
    print(f"Connected to Redis at {args.redis_host}:{args.redis_port}")

    if args.flush:
        keys = list(r.scan_iter(match="cands:*", count=1000))
        if keys:
            r.delete(*keys)
            print(f"Flushed {len(keys):,} existing 'cands:*' keys")

    df = pd.read_parquet(args.input)
    print(f"Loading candidates for {df['user_id'].nunique():,} users ({len(df):,} rows) ...")

    # Group by user first (in memory — this is small, just item_id + score
    # per user, not the full feature set) so each user's candidates go into
    # one ZADD call instead of one call per row.
    by_user = defaultdict(dict)
    for row in df.itertuples(index=False):
        by_user[row.user_id][int(row.cand_item_id)] = float(row.score)

    pipe = r.pipeline(transaction=False)
    for i, (user_id, mapping) in enumerate(tqdm(by_user.items(), total=len(by_user))):
        pipe.delete(f"cands:{user_id}")  # in case of a re-run without --flush
        pipe.zadd(f"cands:{user_id}", mapping)
        if (i + 1) % BATCH_SIZE == 0:
            pipe.execute()
    pipe.execute()

    n_keys = sum(1 for _ in r.scan_iter(match="cands:*", count=1000))
    print(f"Done. Redis now has {n_keys:,} 'cands:*' keys.")
    print("Example lookup: redis-cli ZREVRANGE cands:<some_user_id> 0 9 WITHSCORES")


if __name__ == "__main__":
    main()
