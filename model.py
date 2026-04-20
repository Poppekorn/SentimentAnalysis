"""
Christopher Bryant | ITCS-5154
"""


"""
Models:
  1. LSTMBaseline   - LSTM only, no sentiment
  2. CNNFusion      - LSTM + CNN sentiment scores (Jing et al. 2021)
  3. FinBERTFusion  - LSTM + FinBERT sentiment scores (Catelli et al. 2024)
"""
import torch
import torch.nn as nn
import config

# LSTMBaseline
class LSTMBaseline(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # 2-layer LSTM (Module 9 used single-layer feedforward — this extends to recurrent)
        self.lstm = nn.LSTM(num_features, config.LSTM_H1, num_layers=2,
                           batch_first=True, dropout=config.DROPOUT)
        self.fc = nn.Linear(config.LSTM_H1, 1)

    def forward(self, x_seq, x_sent=None):
        out, _ = self.lstm(x_seq)
        last = out[:, -1, :]              # take last timestep only
        return self.fc(last).squeeze(-1)

# CNNFusion
class CNNFusion(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.lstm = nn.LSTM(num_features, config.LSTM_H1, num_layers=2,
                           batch_first=True, dropout=config.DROPOUT)
        # Small dense net for the 3 sentiment scores (pos/neg/neu)
        self.sent_fc = nn.Sequential(nn.Linear(3, 16), nn.ReLU())
        # Fusion head — combines LSTM output + sentiment
        self.out = nn.Sequential(
            nn.Linear(config.LSTM_H1 + 16, 32), nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, x_seq, x_sent):
        out, _ = self.lstm(x_seq)
        lstm_out = out[:, -1, :]
        sent_out = self.sent_fc(x_sent)
        # Concatenate the two branches (Module 9 lab showed similar concat patterns)
        combined = torch.cat([lstm_out, sent_out], dim=-1)
        return self.out(combined).squeeze(-1)

# FinBERTFusion
class FinBERTFusion(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.lstm = nn.LSTM(num_features, config.LSTM_H1, num_layers=2,
                           batch_first=True, dropout=config.DROPOUT)
        self.sent_fc = nn.Sequential(nn.Linear(3, 16), nn.ReLU())
        self.out = nn.Sequential(
            nn.Linear(config.LSTM_H1 + 16, 32), nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, x_seq, x_sent):
        out, _ = self.lstm(x_seq)
        lstm_out = out[:, -1, :]
        sent_out = self.sent_fc(x_sent)
        combined = torch.cat([lstm_out, sent_out], dim=-1)
        return self.out(combined).squeeze(-1)

# Build by name
def build_model(name, num_features):
    if name == "baseline":
        return LSTMBaseline(num_features)
    elif name == "cnn_fusion":
        return CNNFusion(num_features)
    elif name == "finbert_fusion":
        return FinBERTFusion(num_features)
    else:
        raise ValueError(f"Unknown model: {name}")