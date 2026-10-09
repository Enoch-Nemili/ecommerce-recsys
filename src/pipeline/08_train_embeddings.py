"""
Phase 4a — Train user and item embeddings (two-tower model).

Reuses the same training examples from Phase 2 (real positives from the
future window, popularity-weighted negatives) — same labeled data, but
instead of feeding hand-built features into LightGBM, we let the model
LEARN a vector representation for every user and every item from scratch.

The idea: each user gets a small vector (an "embedding"), each item gets
a small vector, and the model is trained so that a user's vector, dotted
with an item's vector, is high for items they'd actually want and low
otherwise. After training, similar users end up with similar vectors, and
similarly-liked items end up with similar vectors — enabling candidate
generation for items with no co-visitation history at all, purely by
vector similarity (Phase 4b, FAISS).

This is a standard "two-tower" architecture: one small network (tower)
turns a user_id into a vector, another turns an item_id into a vector,
and their dot product is the predicted preference score.

GPU note: this script auto-detects CUDA and uses it if available — this
is the training step that actually benefits from the GPU node, unlike
the Spark/LightGBM steps which are CPU/memory-bound.

Usage:
    python src/pipeline/08_train_embeddings.py \
        --input data/processed/train_examples \
        --output data/processed \
        --embedding-dim 32 \
        --epochs 5
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


class InteractionDataset(Dataset):
    def __init__(self, user_idx, item_idx, labels):
        self.user_idx = torch.tensor(user_idx, dtype=torch.long)
        self.item_idx = torch.tensor(item_idx, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.user_idx[i], self.item_idx[i], self.labels[i]


class TwoTowerModel(nn.Module):
    """Deliberately simple: an embedding table for users, one for items,
    and a dot product between them. No hidden layers — this is closer to
    classic matrix factorization than a deep model, which keeps it fast
    to train and easy to explain, while still learning genuinely useful
    vector representations."""

    def __init__(self, n_users: int, n_items: int, embedding_dim: int):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        # Small init — large random embeddings make early training unstable.
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)

    def forward(self, user_idx, item_idx):
        u = self.user_embedding(user_idx)
        i = self.item_embedding(item_idx)
        return (u * i).sum(dim=1)  # dot product -> raw score (logit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="train_examples parquet from Phase 2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading {args.input} ...")
    df = pd.read_parquet(args.input, columns=["user_id", "item_id", "label"])
    print(f"  {len(df):,} rows, {df['label'].mean():.1%} positive")

    # Map raw user_id/item_id (large, sparse numbers) to dense 0..N-1
    # indices — required for embedding lookup tables, and saved to disk
    # so Phase 4b (FAISS) and the serving layer can map back to real IDs.
    user_ids = sorted(df["user_id"].unique())
    item_ids = sorted(df["item_id"].unique())
    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    item_to_idx = {iid: i for i, iid in enumerate(item_ids)}
    print(f"  {len(user_ids):,} unique users, {len(item_ids):,} unique items")

    user_idx = df["user_id"].map(user_to_idx).to_numpy()
    item_idx = df["item_id"].map(item_to_idx).to_numpy()
    labels = df["label"].to_numpy()

    # Hold out a validation split of individual (user, item) PAIRS — not
    # whole users. This is different from the ranker's user-based split in
    # 03_train_ranker.py, and deliberately so: a LightGBM ranker predicts
    # from generic features, so it can be evaluated on users it never saw.
    # An embedding table cannot — a user's embedding only receives
    # gradient updates on rows where that user_id actually appears in a
    # training batch. Holding out an entire user would leave their
    # embedding at random initialization forever, silently producing
    # garbage vectors for every held-out user in the exported file. So
    # here we hold out random pairs instead: every user and item still
    # appears in training (usually many times), and validation checks
    # whether the model correctly predicts specific withheld interactions
    # — the standard, correct evaluation protocol for this kind of model.
    rng = np.random.default_rng(42)
    n = len(df)
    val_mask = rng.random(n) < 0.1
    train_pos = np.where(~val_mask)[0]
    val_pos = np.where(val_mask)[0]

    train_ds = InteractionDataset(user_idx[train_pos], item_idx[train_pos], labels[train_pos])
    val_ds = InteractionDataset(user_idx[val_pos], item_idx[val_pos], labels[val_pos])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"  train: {len(train_ds):,} rows, val: {len(val_ds):,} rows (held out by pair, not by user)")

    model = TwoTowerModel(len(user_ids), len(item_ids), args.embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    print(f"Training for {args.epochs} epochs (embedding_dim={args.embedding_dim}) ...")
    best_val_auc = -1.0
    best_epoch = -1
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for u, i, y in train_loader:
            u, i, y = u.to(device), i.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(u, i)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / n_batches

        model.eval()
        val_loss_total, val_batches = 0.0, 0
        all_val_logits, all_val_labels = [], []
        with torch.no_grad():
            for u, i, y in val_loader:
                u, i, y = u.to(device), i.to(device), y.to(device)
                logits = model(u, i)
                val_loss_total += loss_fn(logits, y).item()
                val_batches += 1
                all_val_logits.append(logits.cpu().numpy())
                all_val_labels.append(y.cpu().numpy())
        val_loss = val_loss_total / max(val_batches, 1)
        val_auc = roc_auc_score(np.concatenate(all_val_labels), np.concatenate(all_val_logits))

        is_best = val_auc > best_val_auc
        marker = "  <- best so far" if is_best else ""
        print(f"  Epoch {epoch + 1}/{args.epochs} — train loss: {train_loss:.4f}  "
              f"val loss: {val_loss:.4f}  val AUC: {val_auc:.4f}{marker}")

        if is_best:
            # Save embeddings from the best-so-far epoch, not just whatever
            # epoch happens to run last. Training loss alone can't tell you
            # when to stop (it keeps dropping even while the model starts
            # memorizing instead of generalizing, as the epoch-by-epoch
            # val numbers above make visible) — val AUC is what actually
            # tracks whether the embeddings are still improving on data
            # they haven't seen. Overwriting on every improvement means
            # the files on disk always reflect the best checkpoint seen so
            # far, regardless of how many total epochs you run.
            best_val_auc = val_auc
            best_epoch = epoch + 1
            model.eval()
            with torch.no_grad():
                best_user_embeddings = model.user_embedding.weight.cpu().numpy().copy()
                best_item_embeddings = model.item_embedding.weight.cpu().numpy().copy()

    print()
    print(f"Best epoch: {best_epoch} (val AUC {best_val_auc:.4f}) — saving embeddings from this checkpoint.")
    print("If train loss keeps dropping while val loss stops improving or rises, "
          "and/or val AUC plateaus well below train performance, that's overfitting — "
          "the checkpoint saved below is already protected against this, but consider "
          "fewer epochs, a smaller --embedding-dim, or a lower --lr on the next run too.")

    user_embeddings = best_user_embeddings
    item_embeddings = best_item_embeddings

    # Export embeddings as plain numpy arrays + the id mappings needed to
    # translate back to real user_id/item_id later.
    np.save(f"{args.output}/user_embeddings.npy", user_embeddings)
    np.save(f"{args.output}/item_embeddings.npy", item_embeddings)
    with open(f"{args.output}/embedding_id_maps.json", "w") as f:
        json.dump({
            "user_id_to_idx": {str(k): v for k, v in user_to_idx.items()},
            "item_id_to_idx": {str(k): v for k, v in item_to_idx.items()},
        }, f)

    print("Done. Wrote:")
    print(f"  {args.output}/user_embeddings.npy  (shape {user_embeddings.shape})")
    print(f"  {args.output}/item_embeddings.npy  (shape {item_embeddings.shape})")
    print(f"  {args.output}/embedding_id_maps.json")


if __name__ == "__main__":
    main()
