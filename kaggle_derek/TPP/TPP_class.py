import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralTPP(nn.Module):
    """
    Transformer-based sequence encoder with a success classification head.
    No longer a true TPP — the TPP intensity heads (event_linear, time_influence,
    intensity_bias) and the NLL loss have been removed in favor of a pure
    success classifier on pooled transformer hidden states.
    """

    def __init__(
        self,
        num_events: int,
        hidden_dim: int,
        static_dim: int = 0,
        num_layers: int = 2,
        nhead: int = 8,
        max_len: int = 128,
    ):
        super().__init__()
        self.static_dim = static_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        # --- Input projections (all to hidden_dim so they can be summed) ---
        self.event_embedding = nn.Embedding(num_events, hidden_dim, padding_idx=0)
        self.time_proj = nn.Linear(1, hidden_dim)

        # Learned positional encoding (faster and simpler than sinusoidal)
        self.pos_encoding = nn.Embedding(max_len, hidden_dim)

        # --- Transformer encoder with causal masking ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Success classifier (same interface as before) ---
        self.success_head = nn.Sequential(
            nn.Linear((hidden_dim * 3) + static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask so position i can only attend to j <= i."""
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, event_seq: torch.Tensor, delta_seq: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        event_seq : (batch, seq_len)  long  — event type ids (0 = padding)
        delta_seq : (batch, seq_len)  float — log1p(delta_t)

        Returns
        -------
        h : (batch, seq_len, hidden_dim) — transformer hidden states
        """
        batch_size, seq_len = event_seq.shape
        device = event_seq.device

        # Event embedding
        event_emb = self.event_embedding(event_seq)  # (B, L, H)

        # Time delta projection
        time_emb = self.time_proj(delta_seq.unsqueeze(-1))  # (B, L, H)

        # Learned position encoding
        positions = (
            torch.arange(seq_len, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )  # (B, L)
        pos_emb = self.pos_encoding(positions)  # (B, L, H)

        # Sum → transformer (causal mask applied)
        x = event_emb + time_emb + pos_emb
        causal_mask = self._generate_causal_mask(seq_len, device)
        h = self.transformer(x, mask=causal_mask)
        return h

    # ------------------------------------------------------------------
    # Success prediction
    # ------------------------------------------------------------------

    def predict_success_logits_from_hidden(
        self,
        h: torch.Tensor,
        event_seq: torch.Tensor,
        mask: torch.Tensor,
        static_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Aggregate hidden states → success logit."""
        last_real_idx = mask.sum(dim=1).long().clamp(min=1) - 1
        batch_idx = torch.arange(event_seq.shape[0], device=event_seq.device)
        final_h = h[batch_idx, last_real_idx]

        expanded_mask = mask.unsqueeze(-1).bool()
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_h = (h * mask.unsqueeze(-1)).sum(dim=1) / lengths
        max_h = h.masked_fill(~expanded_mask, -1e9).max(dim=1).values

        pooled = torch.cat([final_h, mean_h, max_h], dim=1)

        if self.static_dim:
            if static_features is None:
                raise ValueError("static_features required when static_dim > 0")
            pooled = torch.cat([pooled, static_features], dim=1)

        return self.success_head(pooled).squeeze(1)

    def predict_success_logits(
        self,
        event_seq: torch.Tensor,
        delta_seq: torch.Tensor,
        mask: torch.Tensor,
        static_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.encode(event_seq, delta_seq)
        return self.predict_success_logits_from_hidden(h, event_seq, mask, static_features)