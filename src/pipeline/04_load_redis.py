"""
Phase 2c — Load features into Redis (the online feature store).

Why this step exists: item_features and user_features currently live as
Parquet files. That's fine for offline training, but Parquet is not built
for "give me this one user's features right now, in under a millisecond" —
which is exactly what happens on every real recommendation request. Redis
is an in-memory key-value store built for that: instead of scanning files,
we store each user/item's features under one lookup key ahead of time, so
the Java serving layer (Phase 3) can fetch them with a single, near-instant
Redis GET when a real request comes in.

Key design:
    item:{item_id}  -> Redis hash of item feature fields
    user:{user_id}  -> Redis hash of user feature fields

Usage:
    python src/pipeline/04_load_redis.py \
        --features data/processed \
        --redis-host localhost \
        --redis-port 6379
"""

import argparse

import pandas as pd
import redis
from tqdm import tqdm

BATCH_SIZE = 5000  # rows per Redis pipeline flush


def load_item_features(r: redis.Redis, path: str):
    df = pd.read_parquet(f"{path}/item_features")
    df = df.fillna(0)
    print(f"Loading {len(df):,} item feature rows into Redis ...")

    pipe = r.pipeline(transaction=False)
    for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df))):
        key = f"item:{row.item_id}"
        pipe.hset(key, mapping={
            "total_events": int(row.total_events),
            "buy_count": int(row.buy_count),
            "view_count": int(row.view_count),
            "distinct_users": int(row.distinct_users),
            "popularity_score": float(row.popularity_score),
        })
        if (i + 1) % BATCH_SIZE == 0:
            pipe.execute()
    pipe.execute()  # flush remainder


def load_user_features(r: redis.Redis, path: str):
    df = pd.read_parquet(f"{path}/user_features")
    df = df.fillna(0)
    print(f"Loading {len(df):,} user feature rows into Redis ...")

    pipe = r.pipeline(transaction=False)
    for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df))):
        key = f"user:{row.user_id}"
        pipe.hset(key, mapping={
            "total_events": int(row.total_events),
            "num_sessions": int(row.num_sessions),
            "buy_count": int(row.buy_count),
            "top_category": int(row.top_category) if pd.notna(row.top_category) else 0,
        })
        if (i + 1) % BATCH_SIZE == 0:
            pipe.execute()
    pipe.execute()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="dir with item_features/user_features parquet")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--flush", action="store_true", help="Clear existing item:*/user:* keys before loading")
    args = parser.parse_args()

    r = redis.Redis(host=args.redis_host, port=args.redis_port, db=args.redis_db, decode_responses=True)
    r.ping()
    print(f"Connected to Redis at {args.redis_host}:{args.redis_port}")

    if args.flush:
        for pattern in ("item:*", "user:*"):
            keys = list(r.scan_iter(match=pattern, count=1000))
            if keys:
                r.delete(*keys)
                print(f"Flushed {len(keys):,} existing '{pattern}' keys")

    load_item_features(r, args.features)
    load_user_features(r, args.features)

    n_items = sum(1 for _ in r.scan_iter(match="item:*", count=1000))
    n_users = sum(1 for _ in r.scan_iter(match="user:*", count=1000))
    print(f"Done. Redis now has {n_items:,} item keys and {n_users:,} user keys.")
    print("Example lookup: redis-cli HGETALL item:<some_item_id>")


if __name__ == "__main__":
    main()
