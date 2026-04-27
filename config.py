"""
Christopher Bryant | ITCS-5154
"""
import os
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"

for p in [DATA_RAW, DATA_PROCESSED, CHECKPOINTS, RESULTS]:
    p.mkdir(parents=True, exist_ok=True)

# API keys see .env (.gitignore)
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")

# Stocks to study (https://finance.yahoo.com/quote/%5EVIX/)
TICKERS = ['AAPL','TSLA','NVDA','NFLX','JNJ']
START = "2024-04-23"
END = "2026-04-23"
VIX = "^VIX" 

# Technical indicators (TA-Lib)
INDICATORS = {
    "SMA_5":  ("SMA",  {"timeperiod": 5}),
    "SMA_30": ("SMA",  {"timeperiod": 30}),
    "EMA_12": ("EMA",  {"timeperiod": 12}),
    "EMA_26": ("EMA",  {"timeperiod": 26}),
    "RSI":    ("RSI",  {"timeperiod": 14}),
    "MACD":   ("MACD", {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}),
    "MOM":    ("MOM",  {"timeperiod": 14}),
    "OBV":    ("OBV",  {}),
}

# Sentiment models

# CNN params reference Jing et al. (2021) Section 3.2.1
CNN_EMBED_DIM = 100
CNN_FILTERS = 100
CNN_KERNEL_SIZES = [3, 4, 5]
CNN_DROPOUT = 0.5
CNN_VOCAB_SIZE = 20000

FINBERT_MODEL = "ProsusAI/finbert"

# LSTM training, Module 9 Deep Learning lab
LOOKBACK = 10
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 100
PATIENCE = 10
DROPOUT = 0.2
SEED = 42 

LSTM_H1 = 128
LSTM_H2 = 64

# Train/val/test split
TEST_SPLIT = 0.25
VAL_SPLIT = 0.15

# GPU device, Module 9 lab pattern
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Tickers: {TICKERS}")
    print(f"Date range: {START} to {END}")
    print(f"Alpaca key loaded: {bool(ALPACA_KEY)}")
    print("Folders ready")