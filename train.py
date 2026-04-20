"""
Christopher Bryant | ITCS-5154
"""
"""
Run types see models.
    python train.py --model baseline
    python train.py --model cnn_fusion
    python train.py --model finbert_fusion
"""
import argparse
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import config
from model import build_model

# Reproducibility Module 2, module 9
class StockDataset(Dataset):
    def __init__(self, df, feature_cols, sent_cols):
        self.samples = []
        for _, sub in df.groupby("ticker"):
            sub = sub.sort_values("date").reset_index(drop=True)
            X = sub[feature_cols].values.astype(np.float32)
            S = sub[sent_cols].values.astype(np.float32)
            y = sub["label"].values.astype(np.float32)

            for i in range(config.LOOKBACK, len(sub)):
                self.samples.append((
                    X[i - config.LOOKBACK:i],    # past N days
                    S[i],                        # sentiment today
                    y[i],                        # next-day direction
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        seq, sent, label = self.samples[i]
        return torch.from_numpy(seq), torch.from_numpy(sent), torch.tensor(label)

# split data, Module 5
def split_data(df):
    dates = sorted(df["date"].unique())
    n = len(dates)
    test_start = dates[-int(n * config.TEST_SPLIT)]
    val_start = dates[-int(n * (config.TEST_SPLIT + config.VAL_SPLIT))]

    train = df[df["date"] < val_start]
    val = df[(df["date"] >= val_start) & (df["date"] < test_start)]
    test = df[df["date"] >= test_start]
    return train, val, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                       choices=["baseline", "cnn_fusion", "finbert_fusion"])
    args = parser.parse_args()

    # Reproducibility Module 2, Module 9
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    print(f"\nTraining {args.model} on {config.DEVICE}\n")

    # Load processed dataset
    df = pd.read_parquet(config.DATA_PROCESSED / "dataset.parquet")
    df["date"] = pd.to_datetime(df["date"])

    # Pick sentiment columns based on which model
    sent_cols = ["pos", "neg", "neu"]   # FinBERT scores by default

    if args.model == "cnn_fusion":
        # Swap in CNN scores instead
        cnn = pd.read_parquet(config.DATA_PROCESSED / "news_cnn.parquet")
        cnn["date"] = pd.to_datetime(cnn["date"]).dt.normalize()
        # Keep most-confident headline per (ticker, date)
        cnn["conf"] = cnn[["cnn_neg", "cnn_neu", "cnn_pos"]].max(axis=1)
        best = cnn.groupby(["ticker", "date"])["conf"].idxmax()
        daily = cnn.loc[best, ["ticker", "date", "cnn_pos", "cnn_neg", "cnn_neu"]]
        df = df.merge(daily, on=["ticker", "date"], how="left").fillna(0)
        sent_cols = ["cnn_pos", "cnn_neg", "cnn_neu"]

    # Feature columns = everything except metadata and sentiment
    skip = {"date", "ticker", "label", "next_close",
            "pos", "neg", "neu", "cnn_pos", "cnn_neg", "cnn_neu"}
    feature_cols = [c for c in df.columns if c not in skip]

    # Split + dataloaders
    train_df, val_df, _ = split_data(df)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    train_loader = DataLoader(StockDataset(train_df, feature_cols, sent_cols),
                             batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(StockDataset(val_df, feature_cols, sent_cols),
                           batch_size=config.BATCH_SIZE)

    # Build model + optimizer + loss (Module 9 lab pattern)
    model = build_model(args.model, num_features=len(feature_cols)).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, config.EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0
        for x, s, y in train_loader:
            x, s, y = x.to(config.DEVICE), s.to(config.DEVICE), y.to(config.DEVICE)
            loss = criterion(model(x, s), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate, Module 5
        model.eval()
        val_loss, preds, labels = 0, [], []
        with torch.no_grad():
            for x, s, y in val_loader:
                x, s, y = x.to(config.DEVICE), s.to(config.DEVICE), y.to(config.DEVICE)
                logits = model(x, s)
                val_loss += criterion(logits, y).item()
                preds.extend((torch.sigmoid(logits) > 0.5).long().cpu().tolist())
                labels.extend(y.long().cpu().tolist())

        val_acc = accuracy_score(labels, preds)
        print(f"Epoch {epoch:3d} | train loss {train_loss/len(train_loader):.4f}"
              f" | val loss {val_loss/len(val_loader):.4f} | val acc {val_acc:.3f}")

        history.append({"epoch": epoch,
                       "train_loss": train_loss / len(train_loader),
                       "val_loss": val_loss / len(val_loader),
                       "val_acc": val_acc})

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.CHECKPOINTS / f"{args.model}_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\nStopped early at epoch {epoch}")
                break

    # Save training history
    with open(config.RESULTS / f"{args.model}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone! Saved {args.model}_best.pt")