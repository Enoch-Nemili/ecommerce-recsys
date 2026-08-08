"""
Phase 2a — Build training examples.

Uses the history/future time split from Phase 1 (see cutoff.json) to avoid
leakage: item_features and user_features were computed ONLY from events
before the cutoff ("history"). Positive labels here come ONLY from events
AT OR AFTER the cutoff ("future") — this is what we're actually trying to
predict, using nothing but what was knowable beforehand. Without this
split, a feature like "item_buy_count" would already include the very
purchase the model is being asked to predict, which inflates offline
metrics without the model having learned anything real.

Positive label definition is also tightened to buy/cart only (not plain
page views) — "did they interact with it at all" is too loose to be a
meaningful ranking target when a user may have viewed dozens of items in
one session; buy/cart is a real intent signal.

Negative examples: for each user with history features, sample items they
never touched at ANY point (history or future), weighted toward popular
items from the history window.

Usage:
    python src/pipeline/02_build_training_examples.py \
        --events data/raw/UserBehavior.csv \
        --features data/processed \
        --output data/processed \
        --sample-frac 0.05 \
        --neg-per-user 20
"""

import argparse
import json

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

SCHEMA_COLS = ["user_id", "item_id", "category_id", "behavior_type", "ts"]
POSITIVE_BEHAVIORS = ["buy", "cart"]


def build_spark(app_name: str = "recsys-phase2-examples") -> SparkSession:
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


def read_cutoff(features_dir: str) -> int:
    with open(f"{features_dir}/cutoff.json") as f:
        data = json.load(f)
    return int(data["cutoff_ts"])


def build_positives(events, cutoff_ts: int, known_users):
    """Positive labels: buy/cart events in the FUTURE window, restricted to
    users we actually have history features for (unknown users can't be
    scored — there's nothing to rank them with)."""
    future = events.filter(F.col("ts") >= cutoff_ts)
    future = future.filter(F.col("behavior_type").isin(POSITIVE_BEHAVIORS))
    positives = future.select("user_id", "item_id").distinct()
    positives = positives.join(known_users, on="user_id", how="inner")
    return positives.withColumn("label", F.lit(1))


def build_negatives(events, item_features, known_users, neg_per_user: int, pool_size: int = 5000, seed: int = 42):
    """For each known user, sample neg_per_user items they never touched at
    any point (history or future), weighted toward popular items.

    Scalability note: this deliberately avoids a crossJoin between users
    and the full item catalog (users x items can reach tens of billions of
    rows at real dataset scale). Instead it draws from a bounded pool of
    the top `pool_size` popular items, broadcast as a literal array, and
    shuffles per-user — O(users), not O(users x items)."""
    all_touched = events.select("user_id", "item_id").distinct()  # exclude across BOTH windows

    top_items = (
        item_features.orderBy(F.desc("popularity_score"))
        .select("item_id")
        .limit(pool_size)
        .collect()
    )
    pool_ids = [int(row.item_id) for row in top_items]
    pool_array = F.array(*[F.lit(i) for i in pool_ids])

    oversample = min(neg_per_user * 3, len(pool_ids))
    candidates = (
        known_users.withColumn("shuffled_pool", F.slice(F.shuffle(pool_array), 1, oversample))
        .select("user_id", F.explode("shuffled_pool").alias("item_id"))
    )

    negatives = (
        candidates.alias("c")
        .join(
            all_touched.alias("t"),
            (F.col("c.user_id") == F.col("t.user_id")) & (F.col("c.item_id") == F.col("t.item_id")),
            how="left_anti",
        )
        .withColumn(
            "rn", F.row_number().over(Window.partitionBy("c.user_id").orderBy(F.rand(seed + 1)))
        )
        .filter(F.col("rn") <= neg_per_user)
        .drop("rn")
        .withColumn("label", F.lit(0))
    )
    return negatives


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True)
    parser.add_argument("--features", required=True, help="dir with item_features/user_features/cutoff.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--neg-per-user", type=int, default=20)
    args = parser.parse_args()

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    print("Loading events ...")
    events = load_events(spark, args.events, args.sample_frac)
    events.cache()

    print("Loading precomputed features + cutoff ...")
    item_features = spark.read.parquet(f"{args.features}/item_features")
    user_features = spark.read.parquet(f"{args.features}/user_features")
    cutoff_ts = read_cutoff(args.features)
    print(f"  cutoff_ts = {cutoff_ts} (history < cutoff, future >= cutoff)")

    dupe_count = item_features.groupBy("item_id").count().filter("count > 1").count()
    if dupe_count > 0:
        print(f"  WARNING: {dupe_count} duplicate item_id rows in item_features — re-aggregating")
        item_features = item_features.groupBy("item_id").agg(
            F.sum("total_events").alias("total_events"),
            F.sum("buy_count").alias("buy_count"),
            F.sum("view_count").alias("view_count"),
            F.max("distinct_users").alias("distinct_users"),
            F.sum("popularity_score").alias("popularity_score"),
        )

    known_users = user_features.select("user_id").distinct()

    print("Building positive examples (future buy/cart only) ...")
    positives = build_positives(events, cutoff_ts, known_users)
    print(f"  {positives.count():,} positives")

    print(f"Building negative examples ({args.neg_per_user} per user) ...")
    negatives = build_negatives(events, item_features, known_users, args.neg_per_user)
    print(f"  {negatives.count():,} negatives")

    examples = positives.unionByName(negatives)

    print("Joining features ...")
    examples = examples.join(user_features, on="user_id", how="inner")
    examples = examples.join(
        item_features.select(
            "item_id", "total_events", "buy_count", "view_count",
            "distinct_users", "popularity_score",
        ).withColumnRenamed("total_events", "item_total_events")
         .withColumnRenamed("buy_count", "item_buy_count"),
        on="item_id",
        how="inner",
    )

    print(f"Writing {examples.count():,} training examples ...")
    examples.write.mode("overwrite").parquet(f"{args.output}/train_examples")
    print(f"Done. Output: {args.output}/train_examples")

    spark.stop()


if __name__ == "__main__":
    main()
