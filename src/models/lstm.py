"""
Long Short-Term Memory (LSTM) Architecture for Next-Word Prediction.
Assignment 2 - DLP (CS4045)
Student: Hamza Khurram (i210735)
"""

import torch
import torch.nn as nn

class LSTMLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
        pad_idx: int = 0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # PyTorch requires dropout > 0 only when num_layers > 1
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout
        )

        self.fc_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden_state: tuple = None) -> torch.Tensor:
        """
        x: (batch_size, seq_len)
        Returns: logits of shape (batch_size, vocab_size) for the next word
        """
        embedded = self.embedding(x)                     # (batch_size, seq_len, emb_dim)
        embedded = self.embedding_dropout(embedded)

        out, _ = self.lstm(embedded, hidden_state)       # out: (batch_size, seq_len, hidden_dim)
        last_hidden = out[:, -1, :]                      # (batch_size, hidden_dim)

        last_hidden = self.fc_dropout(last_hidden)
        logits = self.fc(last_hidden)                    # (batch_size, vocab_size)
        return logits
