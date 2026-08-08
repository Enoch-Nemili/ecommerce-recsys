"""
Phase 1 — Offline data pipeline + candidate generation.

Reads the raw Taobao UserBehavior CSV, samples it for fast local iteration,
sessionizes events per user, and builds a simple item-item co-visitation
table as the first candidate-generation source.

Usage:
    python src/pipeline/01_sessionize_and_candidates.py \
        --input data/raw/UserBehavior.csv \
        --output data/processed \
        --sample-frac 0.05

Run on the full dataset (no --sample-frac) once the logic is validated.
"""

import argparse
import json

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

SCHEMA_COLS = ["user_id", "item_id", "category_id", "behavior_type", "ts"]

# Taobao behavior_type values: pv (view), cart, fav (favorite), buy.
# Weight purchases and cart-adds higher than plain views for co-visitation.
BEHAVIOR_WEIGHT = {"pv": 1.0, "fav": 2.0, "cart": 3.0, "buy": 5.0}

SESSION_GAP_SECONDS = 30 * 60  # a new session starts after 30 min of inactivity


def build_spark(app_name: str = "recsys-phase1") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.driver.memory", "6g")
        .getOrCreate()
    )


def load_events(spark: SparkSession, path: str, sample_frac: float | None):
    df = spark.read.csv(path, header=False, inferSchema=True).toDF(*SCHEMA_COLS)
    df = df.withColumn("event_time", F.to_timestamp(F.col("ts")))

    if sample_frac is not None:
        # Sample by user_id (not by row) so a sampled user keeps their full
        # session history instead of getting fragmented events.
        user_ids = df.select("user_id").distinct()
        sampled_users = user_ids.sample(fraction=sample_frac, seed=42)
        df = df.join(sampled_users, on="user_id", how="inner")

    return df


def sessionize(df):
    """Assign a session_id to each event: a new session starts whenever the
    gap since the previous event (per user) exceeds SESSION_GAP_SECONDS."""
    w = Window.partitionBy("user_id").orderBy("event_time")

    df = df.withColumn("prev_time", F.lag("event_time").over(w))
    df = df.withColumn(
        "gap_seconds",
        F.col("event_time").cast("long") - F.col("prev_time").cast("long"),
    )
    df = df.withColumn(
        "new_session_flag",
        F.when(
            F.col("prev_time").isNull() | (F.col("gap_seconds") > SESSION_GAP_SECONDS),
            1,
        ).otherwise(0),
    )
    df = df.withColumn(
        "session_seq", F.sum("new_session_flag").over(w.rowsBetween(Window.unboundedPreceding, 0))
    )
    df = df.withColumn(
        "session_id", F.concat_ws("_", F.col("user_id"), F.col("session_seq"))
    )
    return df.drop("prev_time", "gap_seconds", "new_session_flag", "session_seq")


def build_item_features(df):
    """Popularity and engagement features per item — Phase 2 will load
    these into Redis as the feature store.

    Grouped by item_id only (not item_id + category_id): an item should
    have one stable category, but if the raw data has any noise where the
    same item appears under >1 category, grouping by both would silently
    produce duplicate item_id rows — a fan-out bug waiting to happen in
    any downstream join. We pick the most frequent category per item as
    the representative one instead."""
    weight_expr = F.create_map(
        *[x for kv in BEHAVIOR_WEIGHT.items() for x in (F.lit(kv[0]), F.lit(kv[1]))]
    )
    df = df.withColumn("weight", weight_expr[F.col("behavior_type")])

    cat_counts = df.groupBy("item_id", "category_id").agg(F.count("*").alias("cat_n"))
    w = Window.partitionBy("item_id").orderBy(F.desc("cat_n"))
    primary_category = (
        cat_counts.withColumn("rank", F.row_number().over(w))
        .filter(F.col("rank") == 1)
        .select("item_id", "category_id")
    )

    agg = df.groupBy("item_id").agg(
        F.count("*").alias("total_events"),
        F.sum(F.when(F.col("behavior_type") == "buy", 1).otherwise(0)).alias("buy_count"),
        F.sum(F.when(F.col("behavior_type") == "pv", 1).otherwise(0)).alias("view_count"),
        F.countDistinct("user_id").alias("distinct_users"),
        F.sum("weight").alias("popularity_score"),
    )
    return agg.join(primary_category, on="item_id", how="left")


def build_user_features(df):
    """Basic per-user profile — recency, activity level, category affinity
    (top category by weighted interaction)."""
    weight_expr = F.create_map(
        *[x for kv in BEHAVIOR_WEIGHT.items() for x in (F.lit(kv[0]), F.lit(kv[1]))]
    )
    df = df.withColumn("weight", weight_expr[F.col("behavior_type")])

    user_activity = df.groupBy("user_id").agg(
        F.count("*").alias("total_events"),
        F.countDistinct("session_id").alias("num_sessions"),
        F.max("event_time").alias("last_active"),
        F.sum(F.when(F.col("behavior_type") == "buy", 1).otherwise(0)).alias("buy_count"),
    )

    cat_affinity = (
        df.groupBy("user_id", "category_id")
        .agg(F.sum("weight").alias("cat_weight"))
        .withColumn(
            "rank", F.row_number().over(
                Window.partitionBy("user_id").orderBy(F.desc("cat_weight"))
            ),
        )
        .filter(F.col("rank") == 1)
        .select("user_id", F.col("category_id").alias("top_category"))
    )

    return user_activity.join(cat_affinity, on="user_id", how="left")


def build_covisitation_candidates(df, top_k: int = 50):
    """Item-item co-visitation: for pairs of items seen in the same session,
    count co-occurrences. This is candidate-generation v1 — a fast,
    interpretable baseline that Phase 4 will later replace/augment with
    ANN retrieval over learned embeddings."""
    session_items = df.select("session_id", "item_id").distinct()

    pairs = (
        session_items.alias("a")
        .join(session_items.alias("b"), on="session_id")
        .filter(F.col("a.item_id") != F.col("b.item_id"))
        .groupBy(F.col("a.item_id").alias("item_id"), F.col("b.item_id").alias("candidate_id"))
        .agg(F.count("*").alias("covisit_count"))
    )

    w = Window.partitionBy("item_id").orderBy(F.desc("covisit_count"))
    top_candidates = (
        pairs.withColumn("rank", F.row_number().over(w))
        .filter(F.col("rank") <= top_k)
        .drop("rank")
    )
    return top_candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Fraction of users to sample (e.g. 0.05). Omit to run on all data.",
    )
    parser.add_argument("--top-k-candidates", type=int, default=50)
    parser.add_argument(
        "--history-frac",
        type=float,
        default=0.8,
        help=(
            "Fraction of the time range to treat as 'history' for computing "
            "features. The remaining (1 - history_frac) at the end is left "
            "out as the 'future' window Phase 2 predicts into — this avoids "
            "leaking the label event's own stats into its own features. "
            "Set to 1.0 to disable the split (not recommended for training)."
        ),
    )
    args = parser.parse_args()

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    print(f"Loading events from {args.input} (sample_frac={args.sample_frac}) ...")
    events = load_events(spark, args.input, args.sample_frac)
    events.cache()
    n_users = events.select("user_id").distinct().count()
    print(f"Loaded {events.count():,} events for {n_users:,} users")

    # Defensive: a handful of rows in the raw file can have corrupted
    # timestamps (this showed up in practice — a couple of garbage rows
    # with a negative or far-future ts skewed a naive min/max badly enough
    # to push the whole history/future cutoff into the wrong decade).
    # Fix: find the median timestamp first (a single outlier can't move a
    # median the way it can move a min or max), then drop any row whose ts
    # falls implausibly far from it — 30 days is a generous window for a
    # dataset that only actually spans about 9 days.
    median_ts = events.approxQuantile("ts", [0.5], 0.001)[0]
    plausible_window = 30 * 24 * 3600  # seconds
    events_clean = events.filter(
        (F.col("ts") >= median_ts - plausible_window)
        & (F.col("ts") <= median_ts + plausible_window)
    )
    n_dropped = events.count() - events_clean.count()
    if n_dropped > 0:
        print(f"  Dropped {n_dropped:,} events with implausible timestamps (data quality noise)")
    events = events_clean
    events.cache()

    min_ts, max_ts = events.agg(F.min("ts"), F.max("ts")).first()
    cutoff_ts = int(events.approxQuantile("ts", [args.history_frac], 0.001)[0])
    print(
        f"Time range: {min_ts} - {max_ts}. Using history window ts < {cutoff_ts} "
        f"({args.history_frac:.0%} of events by volume, not by raw time range) "
        f"for features; the rest is held out as the 'future' label window for Phase 2."
    )

    history = events.filter(F.col("ts") < cutoff_ts)
    print(f"History window: {history.count():,} events")

    print("Sessionizing ...")
    sessions = sessionize(history)
    sessions.cache()

    print("Building item features (history only) ...")
    item_features = build_item_features(sessions)
    item_features.write.mode("overwrite").parquet(f"{args.output}/item_features")

    print("Building user features (history only) ...")
    user_features = build_user_features(sessions)
    user_features.write.mode("overwrite").parquet(f"{args.output}/user_features")

    print(f"Building co-visitation candidates (top {args.top_k_candidates} per item, history only) ...")
    candidates = build_covisitation_candidates(sessions, top_k=args.top_k_candidates)
    candidates.write.mode("overwrite").parquet(f"{args.output}/covisit_candidates")

    # Save the cutoff so Phase 2 splits the SAME raw file the same way —
    # otherwise its "future" labels could overlap the events these
    # features were computed from, reintroducing leakage.
    # Written with plain Python (not spark.createDataFrame) — a distributed
    # write is massive overkill for 3 numbers, and on some newer Python
    # versions PySpark's internal pickling of tiny inline dataframes hits a
    # RecursionError. Plain json.dump sidesteps that entirely.
    cutoff_path = f"{args.output}/cutoff.json"
    with open(cutoff_path, "w") as f:
        json.dump({"cutoff_ts": cutoff_ts, "min_ts": int(min_ts), "max_ts": int(max_ts)}, f)

    print("Done. Outputs written to:")
    print(f"  {args.output}/item_features")
    print(f"  {args.output}/user_features")
    print(f"  {args.output}/covisit_candidates")
    print(f"  {cutoff_path} (cutoff_ts={cutoff_ts})")

    spark.stop()


if __name__ == "__main__":
    main()
