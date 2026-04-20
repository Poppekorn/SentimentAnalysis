"""
Christopher Bryant | ITCS-5154
"""

"""
Step 2 of the pipeline. Takes the raw data from get_data.py and:
    1. Adds technical indicators (TA-Lib)
    2. Runs FinBERT sentiment on headlines (GPU)
    3. Merges everything together
    4. Creates next-day direction labels
    5. Normalizes features
    6. Saves the final dataset for modeling
"""
import numpy as np
import pandas as pd
import talib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import config


def add_indicators(df):
    out = df.copy().sort_values("date").reset_index(drop=True)
    close = out["close"]
    volume = out["volume"]

    for name, (func_name, params) in config.INDICATORS.items():
        func = getattr(talib, func_name)
        if func_name == "MACD":
            macd, signal, hist = func(close, **params)
            out["MACD"] = macd
            out["MACD_signal"] = signal
            out["MACD_hist"] = hist
        elif func_name == "OBV":
            out[name] = func(close, volume)
        else:
            out[name] = func(close, **params)

    return out

#Run FinBERT, Module 10 lab (Text/NLP), GPU Module 9
def score_with_finbert(headlines):
    print(f"Loading FinBERT on {config.DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(config.FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.FINBERT_MODEL).to(config.DEVICE)
    model.eval()

    all_probs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(headlines), 16), desc="FinBERT"):
            batch = headlines[i:i+16]
            tokens = tokenizer(batch, padding=True, truncation=True,
                              max_length=128, return_tensors="pt").to(config.DEVICE)
            probs = torch.softmax(model(**tokens).logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    return np.concatenate(all_probs)


def pick_daily_sentiment(news, probs):
    news = news.reset_index(drop=True)
    news["pos"] = probs[:, 0]
    news["neg"] = probs[:, 1]
    news["neu"] = probs[:, 2]
    news["confidence"] = probs.max(axis=1)

    best = news.groupby(["ticker", "date"])["confidence"].idxmax()
    return news.loc[best, ["ticker", "date", "pos", "neg", "neu"]].reset_index(drop=True)


if __name__ == "__main__":
    # 1 Load raw data
    print("Loading raw data   ")
    ohlcv = pd.read_parquet(config.DATA_RAW / "ohlcv.parquet")
    vix = pd.read_parquet(config.DATA_RAW / "vix.parquet")
    news = pd.read_parquet(config.DATA_RAW / "news.parquet")

    # Align Alpaca and yfinance dates, missmatch issue 
    ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.normalize()
    vix["date"] = pd.to_datetime(vix["date"]).dt.normalize()
    news["date"] = pd.to_datetime(news["date"]).dt.normalize()

    # 2 Technical indicators per ticker
    print("Computing indicators   ")
    dfs = []
    for ticker in config.TICKERS:
        sub = ohlcv[ohlcv["ticker"] == ticker]
        if not sub.empty:
            dfs.append(add_indicators(sub))
    features = pd.concat(dfs, ignore_index=True)

    # 3 FinBERT sentiment
    headlines = news["headline"].fillna("").astype(str).tolist()
    probs = score_with_finbert(headlines)
    daily_sent = pick_daily_sentiment(news, probs)

    # Save raw scored news, will use with CNN later
    news["finbert_pos"] = probs[:, 0]
    news["finbert_neg"] = probs[:, 1]
    news["finbert_neu"] = probs[:, 2]
    news.to_parquet(config.DATA_PROCESSED / "news_finbert.parquet", index=False)

    # 4 Merge prices,indicators,VIX,sentiment
    print("Merging  ")
    df = features.merge(vix, on="date", how="left")
    df = df.merge(daily_sent, on=["ticker", "date"], how="left")

    # Fill NA, neutral
    df["pos"] = df["pos"].fillna(0.0)
    df["neg"] = df["neg"].fillna(0.0)
    df["neu"] = df["neu"].fillna(1.0)

    # Fill VIX gaps, closed days
    df = df.sort_values(["ticker", "date"])
    df["vix"] = df.groupby("ticker")["vix"].ffill()

    # 5 Next-day direction labels, Module 2 lab pattern
    df["next_close"] = df.groupby("ticker")["close"].shift(-1)
    df["label"] = (df["next_close"] > df["close"]).astype(int)
    df = df.dropna(subset=["next_close"])

    # dropna missing rows
    indicator_cols = [c for c in df.columns
                      if c in config.INDICATORS or c in ["MACD", "MACD_signal", "MACD_hist"]]
    df = df.dropna(subset=indicator_cols)

    # 6 Normalize numeric features, Module 2 lab pattern
    skip = ["date", "ticker", "label", "next_close", "pos", "neg", "neu"]
    num_cols = [c for c in df.columns if c not in skip]
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # 7 Save final dataset
    df.to_parquet(config.DATA_PROCESSED / "dataset.parquet", index=False)
    print(f"\nDone! {len(df)} rows, label balance: {df['label'].value_counts().to_dict()}")