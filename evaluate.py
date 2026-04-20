"""
Christopher Bryant | ITCS-5154
"""
import argparse
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import config
from model import build_model
from train import StockDataset, split_data


def evaluate_model(model_name):
    print(f"\n=== {model_name} ===")

    # Load data, same pattern as train.py maybe put in its own?
    df = pd.read_parquet(config.DATA_PROCESSED / "dataset.parquet")
    df["date"] = pd.to_datetime(df["date"])

    sent_cols = ["pos", "neg", "neu"]
    if model_name == "cnn_fusion":
        cnn = pd.read_parquet(config.DATA_PROCESSED / "news_cnn.parquet")
        cnn["date"] = pd.to_datetime(cnn["date"]).dt.normalize()
        daily = cnn.groupby(["ticker", "date"])[["cnn_pos", "cnn_neg", "cnn_neu"]].mean().reset_index()
        df = df.merge(daily, on=["ticker", "date"], how="left").fillna(0)
        sent_cols = ["cnn_pos", "cnn_neg", "cnn_neu"]

    skip = {"date", "ticker", "label", "next_close",
            "pos", "neg", "neu", "cnn_pos", "cnn_neg", "cnn_neu"}
    feature_cols = [c for c in df.columns if c not in skip]

    # Get TEST split
    _, _, test_df = split_data(df)
    test_loader = DataLoader(StockDataset(test_df, feature_cols, sent_cols),
                             batch_size=config.BATCH_SIZE)

    # Load trained model
    model = build_model(model_name, len(feature_cols)).to(config.DEVICE)
    model.load_state_dict(torch.load(config.CHECKPOINTS / f"{model_name}_best.pt",
                                     map_location=config.DEVICE))
    model.eval()

    # Predict
    preds, labels = [], []
    with torch.no_grad():
        for x, s, y in test_loader:
            x, s = x.to(config.DEVICE), s.to(config.DEVICE)
            p = (torch.sigmoid(model(x, s)) > 0.5).long().cpu().tolist()
            preds.extend(p)
            labels.extend(y.long().tolist())

    # Metrics (Module 5 lab)
    metrics = {
        "model": model_name,
        "accuracy":  round(accuracy_score(labels, preds), 4),
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall":    round(recall_score(labels, preds, zero_division=0), 4),
        "f1":        round(f1_score(labels, preds, zero_division=0), 4),
    }
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(classification_report(labels, preds, target_names=["Down", "Up"], zero_division=0))

    # Save metrics
    with open(config.RESULTS / f"{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix plot
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Down", "Up"], yticklabels=["Down", "Up"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name}")
    plt.tight_layout()
    plt.savefig(config.RESULTS / f"{model_name}_confusion.png", dpi=150)
    plt.close()

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "cnn_fusion", "finbert_fusion"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        results = []
        for name in ["baseline", "cnn_fusion", "finbert_fusion"]:
            if (config.CHECKPOINTS / f"{name}_best.pt").exists():
                results.append(evaluate_model(name))

        # Comparison table
        comp = pd.DataFrame(results)
        print("\n=== COMPARISON ===")
        print(comp.to_string(index=False))
        comp.to_csv(config.RESULTS / "comparison.csv", index=False)
    elif args.model:
        evaluate_model(args.model)
    else:
        print("Use --model <name> or --all")