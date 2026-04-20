"""
Christopher Bryant | ITCS-5154
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
import config


# Simple word, tokenizer
class Tokenizer:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}

    def fit(self, texts):
        counter = Counter()
        for t in texts:
            counter.update(t.lower().split())
        for word, _ in counter.most_common(20000):
            self.word2idx[word] = len(self.word2idx)

    def encode(self, text, max_len=64):
        ids = [self.word2idx.get(w, 1) for w in text.lower().split()[:max_len]]
        return ids + [0] * (max_len - len(ids))


# TextCNN
class TextCNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 100, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(100, 100, k) for k in [3, 4, 5]])
        self.fc = nn.Linear(300, 3)   # 3 classes: neg, neu, pos

    def forward(self, x):
        e = self.emb(x).permute(0, 2, 1)
        pooled = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.fc(torch.cat(pooled, dim=1))


if __name__ == "__main__":
    # 1 Load Financial PhraseBank (4800+ labeled financial sentences)
    print("Loading Financial PhraseBank ")
    ds = load_dataset("descartes100/enhanced-financial-phrasebank", split="train")
    # This dataset nests rows under 'train' — unwrap them
    rows = [row["train"] for row in ds]
    texts = [r["sentence"] for r in rows]
    labels = [r["label"] for r in rows]
    print(f"Loaded {len(texts)} sentences")
    # 2 Tokenize
    tok = Tokenizer()
    tok.fit(texts)
    X = [tok.encode(t) for t in texts]

    # 3 Train/val split
    X_tr, X_val, y_tr, y_val = train_test_split(X, labels, test_size=0.2, random_state=42)

    # Tensor batches
    tr_x, tr_y = torch.tensor(X_tr), torch.tensor(y_tr)
    va_x, va_y = torch.tensor(X_val), torch.tensor(y_val)
    train_loader = DataLoader(list(zip(tr_x, tr_y)), batch_size=32, shuffle=True)
    val_loader = DataLoader(list(zip(va_x, va_y)), batch_size=32)

    # 4 Train
    model = TextCNN(len(tok.word2idx)).to(config.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\nTraining on {config.DEVICE}...")
    best_acc = 0
    for epoch in range(1, 21):
        model.train()
        for x, y in train_loader:
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Validate
        model.eval()
        preds, truth = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(config.DEVICE)
                preds.extend(model(x).argmax(dim=1).cpu().tolist())
                truth.extend(y.tolist())

        acc = (np.array(preds) == np.array(truth)).mean()
        print(f"  Epoch {epoch:2d} | val acc {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), config.CHECKPOINTS / "cnn_sentiment.pt")

    print(f"\nBest accuracy: {best_acc:.4f}")
    print(classification_report(truth, preds, target_names=["neg", "neu", "pos"]))

    # 5 Score all news headlines with the trained CNN
    print("\nScoring headlines...")
    model.load_state_dict(torch.load(config.CHECKPOINTS / "cnn_sentiment.pt"))
    model.eval()

    news = pd.read_parquet(config.DATA_RAW / "news.parquet")
    headlines = news["headline"].fillna("").astype(str).tolist()

    probs = []
    with torch.no_grad():
        for i in range(0, len(headlines), 32):
            batch = torch.tensor([tok.encode(h) for h in headlines[i:i+32]]).to(config.DEVICE)
            probs.append(torch.softmax(model(batch), dim=-1).cpu().numpy())

    probs = np.concatenate(probs)
    news["cnn_neg"] = probs[:, 0]
    news["cnn_neu"] = probs[:, 1]
    news["cnn_pos"] = probs[:, 2]
    news.to_parquet(config.DATA_PROCESSED / "news_cnn.parquet", index=False)

    print(f"Done! Saved to {config.DATA_PROCESSED / 'news_cnn.parquet'}")