import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Classic sinusoidal positional encoding (Vaswani et al., 2017).

    Adds a fixed pattern to each position so the model can distinguish
    where in the sequence each token came from — without recurrence.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                          # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()    # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                                        # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional encoding, same shape
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Input projection
# ---------------------------------------------------------------------------

class InputProjection(nn.Module):
    """
    Projects raw features to d_model dimensions.
    Separates feature engineering from the Transformer core.
    """

    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        self.projection = nn.Linear(num_features, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seq_len, num_features)
        Returns:
            (batch, seq_len, d_model)
        """
        return self.norm(self.projection(x))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    """
    Stack of N identical encoder layers, each containing:
      1. Multi-head self-attention
      2. Add & Norm (residual connection + layer norm)
      3. Position-wise feed-forward network
      4. Add & Norm

    The encoder sees the full input sequence at once (bidirectional).
    Ideal for classification, regression, and as the first half of a
    sequence-to-sequence model.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: str = "gelu",
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,   # (batch, seq, features) — natural for time series
            norm_first=True,    # Pre-LN: more stable training than Post-LN
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

    def forward(self, x: Tensor, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x:                    (batch, seq_len, d_model)
            src_key_padding_mask: (batch, seq_len) bool, True = masked/padding position
        Returns:
            (batch, seq_len, d_model)
        """
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

# ---------------------------------------------------------------------------
# Classification Head
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """CLS-token style: uses the representation of the first position."""

    def __init__(self, d_model: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        cls = x[:, 0, :]             # (batch, d_model) — first position
        return self.mlp(cls)         # (batch, num_classes)