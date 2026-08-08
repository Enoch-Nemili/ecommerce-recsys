"""
Phase 2b — Train and evaluate the ranking model.

Reads data/processed/train_examples (built by 02_build_training_examples.py),
trains a LightGBM binary classifier (label = did the user interact with
this item), and evaluates with:
  - AUC: can the model separate positives from negatives at all?
  - Recall@K / NDCG@K: for each user, out of their candidates, does the
    model actually rank their real positive item near the top?

Usage:
    python src/pipeline/03_train_ranker.py \
        --input data/processed/train_examples \
        --model-output data/processed/ranker_model.txt
"""

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

FEATURE_COLS = [
    "total_events", "num_sessions", "buy_count",       # user features
    "item_total_events", "item_buy_count", "view_count",
    "distinct_users", "popularity_score",               # item features
]


def load_examples(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.dropna(subset=FEATURE_COLS + ["label"])
    return df


def split_train_test(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Split by user_id, not by row — if the same user's examples ended up
    in both train and test, the model could partly memorize that user
    instead of generalizing, and offline metrics would look better than
    they'd actually perform on a new user."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df["user_id"]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def train_model(train_df: pd.DataFrame):
    X = train_df[FEATURE_COLS]
    y = train_df["label"]
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=5,
        verbosity=-1,
    )
    model.fit(X, y)
    return model


def evaluate_auc(model, test_df: pd.DataFrame) -> float:
    X = test_df[FEATURE_COLS]
    y = test_df["label"]
    preds = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, preds)


def evaluate_ranking(model, test_df: pd.DataFrame, k: int = 10):
    """Recall@K and NDCG@K, computed per user: score every candidate item
    in the test set for that user, rank them, and check whether the true
    positive(s) landed in the top K."""
    X = test_df[FEATURE_COLS]
    test_df = test_df.copy()
    test_df["score"] = model.predict_proba(X)[:, 1]

    recalls, ndcgs = [], []
    for _, group in test_df.groupby("user_id"):
        if group["label"].sum() == 0:
            continue  # no positive to find for this user in the test split
        ranked = group.sort_values("score", ascending=False).reset_index(drop=True)
        top_k = ranked.head(k)

        hits = top_k["label"].sum()
        total_pos = group["label"].sum()
        recalls.append(hits / total_pos)

        # NDCG@K: reward positives found earlier in the ranked list more
        # than positives found later.
        gains = top_k["label"].values
        discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
        dcg = float(np.sum(gains * discounts))
        ideal_gains = np.sort(group["label"].values)[::-1][:k]
        idcg = float(np.sum(ideal_gains * discounts[: len(ideal_gains)]))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(recalls)), float(np.mean(ndcgs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-output", default=None)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    print(f"Loading training examples from {args.input} ...")
    df = load_examples(args.input)
    print(f"  {len(df):,} examples, {df['label'].mean():.1%} positive")

    train_df, test_df = split_train_test(df)
    print(f"  train: {len(train_df):,} rows / {train_df['user_id'].nunique():,} users")
    print(f"  test:  {len(test_df):,} rows / {test_df['user_id'].nunique():,} users")

    print("Training LightGBM ranker ...")
    model = train_model(train_df)

    auc = evaluate_auc(model, test_df)
    recall_k, ndcg_k = evaluate_ranking(model, test_df, k=args.k)

    print()
    print("=== Evaluation ===")
    print(f"AUC:          {auc:.4f}")
    print(f"Recall@{args.k}:    {recall_k:.4f}")
    print(f"NDCG@{args.k}:      {ndcg_k:.4f}")

    print()
    print("Feature importances:")
    for name, imp in sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:<20s} {imp}")

    if args.model_output:
        model.booster_.save_model(args.model_output)
        print(f"\nModel saved to {args.model_output}")


if __name__ == "__main__":
    main()
