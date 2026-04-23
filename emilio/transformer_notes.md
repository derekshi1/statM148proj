# Sequential Processing Example

Journey (variable length sequence)
    │
    ▼  InputProjection
Each step's features → d_model-dim vector

    │
    ▼  PositionalEncoding
+ signal encoding which step came first, second, etc.

    │
    ▼  Encoder (× N layers)
Each position attends to every other position.
Isolated features → contextual representations h₁ … hₜ

    │
    ▼  Mean Pool
(h₁ + h₂ + … + hₜ) / T  →  one vector summarising the whole journey

    │
    ▼  MLP head
Summary vector → logits for each class

    │
    ▼  Softmax
[0.05,  0.72,  0.12,  0.08,  0.03]
         ↑
    predicted class