"""
Phase 3a — Precompute per-user candidate lists.

Why this exists: co-visitation tells us "item X is similar to item Y," but
the live service needs to answer "what should I show user 123" fast. Doing
that lookup live (find user's history -> look up covisitation for each of
their items -> merge -> dedupe) is exactly the kind of multi-step, data-
heavy work that belongs in an offline batch job, not in the hot path of a
web request. This script does that work once, in Spark, and produces a
simple user_id -> [candidate item_ids] table that Redis can serve instantly.

For each user (from the history window): take their most-interacted items,
look up each one's co-visitation candidates, merge and rank by total
co-visitation weight, exclude items they've already touched, keep the
top N.

Usage:
    python src/pipeline/06_build_user_candidates.py \
        --events data/raw/UserBehavior.csv \
        --features data/processed \
        --output data/processed \
        --sample-frac 0.05 \
        --top-n 50
"""

import argparse
import json

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

SCHEMA_COLS = ["user_id", "item_id", "category_id", "behavior_type", "ts"]


def build_spark(app_name: str = "recsys-phase3-candidates") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.driver.memory", "6g")
        .getOrCreate()
    )


def load_events(spark, path, sample_frac):
    df = spark.read.csv(path, header=False, inferSchema=True).toDF(*SCHEMA_COLS)
    if sample_frac is not None:
        user_ids = df.select("user_id").distinct()
        sampled_users = user_ids.sample(fraction=sample_frac, seed=42)
        df = df.join(sampled_users, on="user_id", how="inner")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True)
    parser.add_argument("--features", required=True, help="dir with covisit_candidates, cutoff.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--seed-items-per-user", type=int, default=10,
                         help="How many of a user's own most-recent history items to use as seeds")
    args = parser.parse_args()

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    with open(f"{args.features}/cutoff.json") as f:
        cutoff_ts = int(json.load(f)["cutoff_ts"])

    print("Loading events ...")
    events = load_events(spark, args.events, args.sample_frac)
    history = events.filter(F.col("ts") < cutoff_ts)

    print(f"Loading co-visitation candidates from {args.features}/covisit_candidates ...")
    covisit = spark.read.parquet(f"{args.features}/covisit_candidates")

    # Seed items: each user's most recent N distinct items from their history.
    w = Window.partitionBy("user_id").orderBy(F.desc("ts"))
    seeds = (
        history.withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") <= args.seed_items_per_user)
        .select("user_id", "item_id")
        .distinct()
    )
    print(f"  {seeds.count():,} (user, seed_item) pairs")

    # Everything the user has ever touched (history + future) — exclude
    # these from their own candidate list, there's no point recommending
    # something they've already interacted with.
    already_touched = events.select("user_id", "item_id").distinct()

    # Expand seeds -> co-visitation candidates, merge by summing weight
    # across all of a user's seed items (an item that co-occurs with
    # several of the user's own items is a stronger candidate).
    expanded = seeds.join(covisit, on="item_id", how="inner").select(
        "user_id", F.col("candidate_id").alias("cand_item_id"), "covisit_count"
    )

    merged = expanded.groupBy("user_id", "cand_item_id").agg(
        F.sum("covisit_count").alias("score")
    )

    merged = merged.alias("m").join(
        already_touched.alias("t"),
        (F.col("m.user_id") == F.col("t.user_id")) & (F.col("m.cand_item_id") == F.col("t.item_id")),
        how="left_anti",
    )

    rank_w = Window.partitionBy("user_id").orderBy(F.desc("score"))
    top_candidates = (
        merged.withColumn("rank", F.row_number().over(rank_w))
        .filter(F.col("rank") <= args.top_n)
        .select("user_id", "cand_item_id", "score", "rank")
    )

    print(f"Writing top-{args.top_n} candidates per user ...")
    top_candidates.write.mode("overwrite").parquet(f"{args.output}/user_candidates")

    n_users_with_candidates = top_candidates.select("user_id").distinct().count()
    print(f"Done. {n_users_with_candidates:,} users have precomputed candidates.")
    print(f"Output: {args.output}/user_candidates")

    spark.stop()


if __name__ == "__main__":
    main()
