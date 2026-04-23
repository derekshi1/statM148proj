"""
Time Series Transformer
=======================
A general-purpose Transformer architecture for time series tasks.

Supports three task modes:
  - 'forecasting'    : encoder-decoder, predicts future horizon
  - 'regression'     : encoder-only, maps sequence → scalar(s)
  - 'classification' : encoder-only, maps sequence → class logits

Input shape:  (batch, seq_len, num_features)
Output shape: depends on task mode (see TimeSeriesTransformer.forward docstring)
"""

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
# Decoder (for forecasting)
# ---------------------------------------------------------------------------

class TransformerDecoder(nn.Module):
    """
    Stack of N identical decoder layers, each containing:
      1. Masked multi-head self-attention  (causal — can't look at future targets)
      2. Add & Norm
      3. Cross-attention over encoder output  (queries from decoder, keys/values from encoder)
      4. Add & Norm
      5. Feed-forward network
      6. Add & Norm

    Used for autoregressive forecasting: the decoder generates the output
    sequence one step at a time (during inference) or all at once with
    causal masking (during training — teacher forcing).
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
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            tgt:                      (batch, tgt_len, d_model) — decoder input
            memory:                   (batch, src_len, d_model) — encoder output
            tgt_mask:                 (tgt_len, tgt_len) causal mask
            tgt_key_padding_mask:     (batch, tgt_len)
            memory_key_padding_mask:  (batch, src_len)
        Returns:
            (batch, tgt_len, d_model)
        """
        return self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )


# ---------------------------------------------------------------------------
# Output heads
# ---------------------------------------------------------------------------

class ForecastingHead(nn.Module):
    """Projects decoder output to predicted feature values."""

    def __init__(self, d_model: int, num_output_features: int):
        super().__init__()
        self.proj = nn.Linear(d_model, num_output_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)   # (batch, horizon, num_output_features)


class RegressionHead(nn.Module):
    """
    Global average-pool encoder output, then project to output_dim.
    For tasks like predicting a single scalar from a window.
    """

    def __init__(self, d_model: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        pooled = x.mean(dim=1)       # (batch, d_model)
        return self.mlp(pooled)      # (batch, output_dim)


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


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class TimeSeriesTransformer(nn.Module):
    """
    General-purpose Transformer for time series.

    Parameters
    ----------
    num_features : int
        Number of input channels/variables per time step.
    d_model : int
        Internal embedding dimension. All attention and FFN layers operate
        at this width. Must be divisible by nhead.
    nhead : int
        Number of attention heads. Each head uses d_model // nhead dimensions.
    num_encoder_layers : int
        Depth of the encoder stack.
    num_decoder_layers : int
        Depth of the decoder stack (only used when task='forecasting').
    dim_feedforward : int
        Width of the inner FFN layer (typically 2×–4× d_model).
    dropout : float
        Dropout applied throughout (attention weights, FFN, embeddings).
    task : str
        One of 'forecasting', 'regression', 'classification'.
    horizon : int
        Number of future steps to predict (forecasting only).
    num_output_features : int
        Variables to forecast (forecasting) or scalar outputs (regression).
    num_classes : int
        Number of classes (classification only).
    max_seq_len : int
        Maximum sequence length for positional encoding.
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        task: str = "forecasting",
        horizon: int = 24,
        num_output_features: int = 1,
        num_classes: int = 2,
        max_seq_len: int = 5000,
    ):
        super().__init__()
        assert task in ("forecasting", "regression", "classification"), \
            f"task must be one of 'forecasting', 'regression', 'classification', got '{task}'"
        assert d_model % nhead == 0, \
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.task = task
        self.d_model = d_model
        self.horizon = horizon

        # --- Shared components ---
        self.input_proj = InputProjection(num_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        self.encoder = TransformerEncoder(
            d_model, nhead, num_encoder_layers, dim_feedforward, dropout
        )

        # --- Task-specific components ---
        if task == "forecasting":
            self.target_proj = InputProjection(num_output_features, d_model)
            self.decoder = TransformerDecoder(
                d_model, nhead, num_decoder_layers, dim_feedforward, dropout
            )
            self.head = ForecastingHead(d_model, num_output_features)

        elif task == "regression":
            self.head = RegressionHead(d_model, num_output_features)

        elif task == "classification":
            self.head = ClassificationHead(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _causal_mask(size: int, device: torch.device) -> Tensor:
        """Upper-triangular mask: prevents decoder from attending to future positions."""
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    def forward(
        self,
        src: Tensor,
        tgt: Optional[Tensor] = None,
        src_padding_mask: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args
        ----
        src : (batch, src_len, num_features)
            The input (past) sequence.
        tgt : (batch, tgt_len, num_output_features)  [forecasting only]
            Target sequence for teacher-forced training. During inference,
            pass a single "start token" (e.g. the last src value) and call
            forward() autoregressively.
        src_padding_mask : (batch, src_len) bool
            True = padding position to ignore (e.g. for variable-length batches).
        tgt_padding_mask : (batch, tgt_len) bool  [forecasting only]

        Returns
        -------
        forecasting    : (batch, horizon, num_output_features)
        regression     : (batch, output_dim)
        classification : (batch, num_classes)
        """
        # --- Encode ---
        x = self.pos_enc(self.input_proj(src))
        memory = self.encoder(x, src_key_padding_mask=src_padding_mask)

        # --- Decode / head ---
        if self.task == "forecasting":
            assert tgt is not None, "tgt must be provided for forecasting task"
            tgt_emb = self.pos_enc(self.target_proj(tgt))
            causal_mask = self._causal_mask(tgt.size(1), src.device)
            dec_out = self.decoder(
                tgt_emb,
                memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask,
            )
            return self.head(dec_out)           # (batch, tgt_len, num_output_features)

        else:
            return self.head(memory)            # (batch, output_dim) or (batch, num_classes)

    @torch.no_grad()
    def forecast(self, src: Tensor, start_token: Tensor) -> Tensor:
        """
        Autoregressive inference for forecasting.

        Generates `self.horizon` steps one at a time, feeding each
        prediction back as the next decoder input.

        Args:
            src:         (batch, src_len, num_features) — past context
            start_token: (batch, 1, num_output_features) — seed (e.g. last observed value)
        Returns:
            (batch, horizon, num_output_features)
        """
        assert self.task == "forecasting", "forecast() is only available for task='forecasting'"
        self.eval()

        x = self.pos_enc(self.input_proj(src))
        memory = self.encoder(x)

        outputs = []
        decoder_input = start_token              # (batch, 1, num_output_features)

        for _ in range(self.horizon):
            tgt_emb = self.pos_enc(self.target_proj(decoder_input))
            causal_mask = self._causal_mask(decoder_input.size(1), src.device)
            dec_out = self.decoder(tgt_emb, memory, tgt_mask=causal_mask)
            next_pred = self.head(dec_out[:, -1:, :])   # (batch, 1, num_output_features)
            outputs.append(next_pred)
            decoder_input = torch.cat([decoder_input, next_pred], dim=1)

        return torch.cat(outputs, dim=1)         # (batch, horizon, num_output_features)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def make_forecasting_transformer(
    num_features: int,
    num_output_features: int = 1,
    horizon: int = 24,
    d_model: int = 128,
    nhead: int = 8,
    num_encoder_layers: int = 4,
    num_decoder_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
) -> TimeSeriesTransformer:
    """Encoder-decoder Transformer for time series forecasting."""
    return TimeSeriesTransformer(
        num_features=num_features,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        task="forecasting",
        horizon=horizon,
        num_output_features=num_output_features,
    )


def make_regression_transformer(
    num_features: int,
    output_dim: int = 1,
    d_model: int = 128,
    nhead: int = 8,
    num_encoder_layers: int = 4,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
) -> TimeSeriesTransformer:
    """Encoder-only Transformer for sequence-to-scalar regression."""
    return TimeSeriesTransformer(
        num_features=num_features,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        task="regression",
        num_output_features=output_dim,
    )


def make_classification_transformer(
    num_features: int,
    num_classes: int,
    d_model: int = 128,
    nhead: int = 8,
    num_encoder_layers: int = 4,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
) -> TimeSeriesTransformer:
    """Encoder-only Transformer for sequence classification."""
    return TimeSeriesTransformer(
        num_features=num_features,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        task="classification",
        num_classes=num_classes,
    )


# ---------------------------------------------------------------------------
# Example usage & smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    BATCH, SRC_LEN, NUM_FEATURES = 16, 96, 7   # e.g. ETTh1 dataset dims
    HORIZON = 24
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- 1. Forecasting ---
    print("=== Forecasting ===")
    model = make_forecasting_transformer(
        num_features=NUM_FEATURES,
        num_output_features=1,
        horizon=HORIZON,
    ).to(device)

    src = torch.randn(BATCH, SRC_LEN, NUM_FEATURES, device=device)
    tgt = torch.randn(BATCH, HORIZON, 1, device=device)  # teacher-forced targets

    out = model(src, tgt)
    print(f"  Training output:   {out.shape}")           # (16, 24, 1)

    start = src[:, -1:, :1]                              # use last observed value
    pred = model.forecast(src, start_token=start)
    print(f"  Autoregressive:    {pred.shape}")          # (16, 24, 1)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:        {total_params:,}\n")

    # --- 2. Regression ---
    print("=== Regression ===")
    reg_model = make_regression_transformer(
        num_features=NUM_FEATURES, output_dim=1
    ).to(device)

    reg_out = reg_model(src)
    print(f"  Output:            {reg_out.shape}")       # (16, 1)

    # --- 3. Classification ---
    print("=== Classification ===")
    clf_model = make_classification_transformer(
        num_features=NUM_FEATURES, num_classes=5
    ).to(device)

    clf_out = clf_model(src)
    print(f"  Logits:            {clf_out.shape}")       # (16, 5)

    # --- Minimal training loop sketch ---
    print("\n=== Training loop sketch ===")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    model.train()
    for step in range(3):
        src_b = torch.randn(BATCH, SRC_LEN, NUM_FEATURES, device=device)
        tgt_b = torch.randn(BATCH, HORIZON, 1, device=device)
        labels = tgt_b[:, 1:, :]                        # predict next step (shift by 1)
        preds  = model(src_b, tgt_b)[:, :-1, :]        # drop last pred to align

        loss = F.mse_loss(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        print(f"  Step {step+1}  loss={loss.item():.4f}")

    print("\nAll checks passed.")