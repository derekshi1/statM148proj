import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


# ---------------------------------------------------------------------------
# Time-relative bias
# ---------------------------------------------------------------------------

class TimeRelativeBias(nn.Module):
    """
    Learns a per-head attention bias from the pairwise time delta between
    every pair of positions (i, j) in the sequence.

    dt[i, j] = timestamp[i] - timestamp[j]

    A small MLP maps this scalar to one bias value per attention head.
    The bias is added to the raw attention scores before softmax, so:
      - Positive bias  → head is encouraged to attend across that time gap
      - Negative bias  → head is discouraged from attending across that gap

    The model learns which gaps matter and in which direction, per head.
    """

    def __init__(self, nhead: int):
        super().__init__()
        self.nhead = nhead
        self.bias_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, nhead),
        )

    def forward(self, timestamps: Tensor) -> Tensor:
        """
        Args:
            timestamps: (batch, seq_len) — cumulative seconds from journey start
        Returns:
            bias: (batch, nhead, seq_len, seq_len)
        """
        # Pairwise time differences: dt[b, i, j] = t[b, i] - t[b, j]
        dt = timestamps.unsqueeze(2) - timestamps.unsqueeze(1)   # (B, T, T)

        # Log-scale the magnitude, preserve sign.
        # Raw seconds can be e.g. 43200 (12 hours), which causes large MLP
        # activations and unstable gradients. log1p(|dt|) compresses the
        # range to ~[0, 11] while keeping 0-delta exactly at 0.
        dt = torch.sign(dt) * torch.log1p(dt.abs())

        dt = dt.unsqueeze(-1).float()                            # (B, T, T, 1)

        # MLP maps each scalar dt to nhead bias values
        bias = self.bias_net(dt)                                 # (B, T, T, nhead)

        # Reorder to match attention score layout (B, nhead, T, T)
        return bias.permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Time-aware self-attention
# ---------------------------------------------------------------------------

class TimeAwareAttention(nn.Module):
    """
    Multi-head self-attention with time-relative positional bias.

    Replaces sinusoidal positional encoding entirely. Instead of adding a
    fixed pattern to the input embeddings, we inject a learned, data-driven
    bias directly into the attention matrix based on the actual timestamps.

    This is strictly more expressive for irregular time series because:
      - Sinusoidal PE assumes uniform step sizes (it encodes position index, not time)
      - TimeRelativeBias encodes actual elapsed time between events
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.qkv      = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout  = nn.Dropout(dropout)
        self.time_bias = TimeRelativeBias(nhead)

    def forward(
        self,
        x: Tensor,
        timestamps: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x:            (batch, seq_len, d_model)
            timestamps:   (batch, seq_len)
            padding_mask: (batch, seq_len) bool — True = padding, ignore position
        Returns:
            (batch, seq_len, d_model)
        """
        B, T, C = x.shape

        # Project to Q, K, V and split into heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.nhead, self.d_k).transpose(1, 2)  # (B, h, T, d_k)
        k = k.view(B, T, self.nhead, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.d_k).transpose(1, 2)

        # Scaled dot-product attention scores
        attn = (q @ k.transpose(-2, -1)) * (self.d_k ** -0.5)   # (B, h, T, T)

        # Add time-relative bias — this is the core modification
        attn = attn + self.time_bias(timestamps)                 # (B, h, T, T)

        # Mask padding positions (set to -inf so softmax → 0)
        if padding_mask is not None:
            # padding_mask: (B, T) → (B, 1, 1, T) for broadcasting over heads and queries
            attn = attn.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = F.softmax(attn, dim=-1)

        # Guard against NaN: if an entire row is -inf (fully padded sequence),
        # softmax produces NaN. Replace with 0 so those positions contribute
        # nothing to the weighted sum rather than poisoning the batch.
        attn = torch.nan_to_num(attn, nan=0.0)

        attn = self.dropout(attn)

        # Weighted sum of values, merge heads
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)       # (B, T, d_model)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Encoder layer (wraps TimeAwareAttention + FFN + residuals)
# ---------------------------------------------------------------------------

class TimeAwareEncoderLayer(nn.Module):
    """
    One encoder layer using TimeAwareAttention instead of standard MHA.

    Structure (Pre-LN):
      x → LayerNorm → TimeAwareAttention → + x  (residual)
        → LayerNorm → FFN                → + x  (residual)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.attn     = TimeAwareAttention(d_model, nhead, dropout)
        self.norm1    = nn.LayerNorm(d_model)
        self.norm2    = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(
        self,
        x: Tensor,
        timestamps: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Pre-LN attention block
        x = x + self.dropout(
            self.attn(self.norm1(x), timestamps, padding_mask)
        )
        # Pre-LN FFN block
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Encoder stack
# ---------------------------------------------------------------------------

class TimeAwareEncoder(nn.Module):
    """Stack of N TimeAwareEncoderLayers."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TimeAwareEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        timestamps: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, timestamps, padding_mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Input projection
# ---------------------------------------------------------------------------

class InputProjection(nn.Module):
    """Projects raw features → d_model with LayerNorm."""

    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        self.projection = nn.Linear(num_features, d_model)
        self.norm       = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.projection(x))


# ---------------------------------------------------------------------------
# Classification head — mean pooling (no CLS token needed)
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """
    Mean-pools the encoder output across all (non-padding) positions,
    then maps to class logits via a two-layer MLP.

    Mean pooling is preferred over CLS-token here because:
      - No CLS token is prepended in the dataset pipeline
      - Mean pooling gives equal weight to all real events by default
      - The model still learns which events matter via attention
    """

    def __init__(self, d_model: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x:            (batch, seq_len, d_model)
            padding_mask: (batch, seq_len) bool — True = padding, exclude from mean
        Returns:
            (batch, num_classes)
        """
        if padding_mask is not None:
            # Zero out padding positions before averaging
            real_mask = (~padding_mask).unsqueeze(-1).float()   # (B, T, 1)  1 = real
            x = x * real_mask
            pooled = x.sum(dim=1) / real_mask.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)

        return self.mlp(pooled)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class TimeSeriesTransformer(nn.Module):
    """
    Time-aware Transformer for journey classification.

    Key differences from the sinusoidal version:
      - No PositionalEncoding module
      - Expects a `timestamps` tensor alongside `src`
      - Uses TimeAwareEncoder (custom layers) instead of nn.TransformerEncoder
      - ClassificationHead uses mean pooling, not CLS token
    """

    def __init__(
        self,
        num_features: int,
        d_model: int        = 128,
        nhead: int          = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float      = 0.1,
        num_classes: int    = 2,
    ):
        super().__init__()
        assert d_model % nhead == 0, \
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.input_proj = InputProjection(num_features, d_model)
        self.encoder    = TimeAwareEncoder(
            d_model, nhead, num_encoder_layers, dim_feedforward, dropout
        )
        self.head = ClassificationHead(d_model, num_classes)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: Tensor,
        timestamps: Tensor,
        src_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            src:             (batch, seq_len, num_features)
            timestamps:      (batch, seq_len) — event times in seconds (or normalised)
            src_padding_mask:(batch, seq_len) bool — True = padding position

        Returns:
            logits: (batch, num_classes)
        """
        x      = self.input_proj(src)                                    # (B, T, d_model)
        memory = self.encoder(x, timestamps, src_padding_mask)           # (B, T, d_model)
        return  self.head(memory, src_padding_mask)                      # (B, num_classes)


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------

def make_classification_transformer(
    num_features: int,
    num_classes: int    = 2,
    d_model: int        = 128,
    nhead: int          = 8,
    num_encoder_layers: int = 4,
    dim_feedforward: int = 256,
    dropout: float      = 0.1,
) -> TimeSeriesTransformer:
    """Encoder-only time-aware Transformer for sequence classification."""
    return TimeSeriesTransformer(
        num_features=num_features,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_classes=num_classes,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, T, F = 8, 64, 8
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model  = make_classification_transformer(num_features=F, num_classes=2).to(device)
    src    = torch.randn(B, T, F, device=device)
    ts     = torch.linspace(0, 3600, T).unsqueeze(0).expand(B, -1).to(device)
    mask   = torch.zeros(B, T, dtype=torch.bool, device=device)
    mask[:, 50:] = True   # last 14 positions are padding

    logits = model(src, timestamps=ts, src_padding_mask=mask)
    print(f"Output shape: {logits.shape}")   # (8, 2)
    print(f"Parameters:   {sum(p.numel() for p in model.parameters()):,}")
    print("Smoke test passed.")