"""
TPP Legacy Model Predictions
Uses the old RNN-based architecture matching saved checkpoints (tpp_model_best.pth)
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from TPP_features import build_static_features

MAX_LEN = 128

MODEL_PATH = SCRIPT_DIR / "tpp_model_best.pth"
TEST_DATA_PATH = ROOT / "kaggle_derek" / "open_journeys2_correct.parquet"
PROCESSED_DATA_PATH = ROOT / "data" / "tpp_processed_data.npz"
OUTPUT_CSV = ROOT / "kaggle_derek" / "tpp_legacy_predictions.csv"
RAW_OUTPUT_CSV = ROOT / "TPP" / "tpp_legacy_raw_predictions.csv"

TIER_1_THRESHOLD = 0.40
TIER_2_THRESHOLD = 0.60
TIER_3_THRESHOLD = 0.75


class LegacyNeuralTPP(nn.Module):
    """Matches old RNN-based architecture from saved checkpoints."""
    def __init__(self, num_events=30, hidden_dim=96, embedding_dim=32, static_dim=38):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.static_dim = static_dim
        self.embedding = nn.Embedding(num_events, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim + 1, hidden_dim, batch_first=True)
        self.event_linear = nn.Linear(hidden_dim, num_events)
        self.time_influence = nn.Linear(hidden_dim, 1)
        self.intensity_bias = nn.Linear(hidden_dim, 1)
        self.success_head = nn.Sequential(
            nn.Linear((hidden_dim * 3) + static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, event_seq, delta_t_seq):
        event_emb = self.embedding(event_seq)
        delta_input = delta_t_seq.unsqueeze(-1)
        rnn_input = torch.cat([event_emb, delta_input], dim=-1)
        h, _ = self.rnn(rnn_input)
        return h

    def predict_success_logits(self, event_seq, delta_t_seq, mask, static_features):
        h = self.encode(event_seq, delta_t_seq)
        last_real_idx = mask.sum(dim=1).long().clamp(min=1) - 1
        batch_idx = torch.arange(event_seq.shape[0], device=event_seq.device)
        final_h = h[batch_idx, last_real_idx]
        expanded_mask = mask.unsqueeze(-1).bool()
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_h = (h * mask.unsqueeze(-1)).sum(dim=1) / lengths
        max_h = h.masked_fill(~expanded_mask, -1e9).max(dim=1).values
        pooled = torch.cat([final_h, mean_h, max_h], dim=1)
        if self.static_dim and static_features is not None:
            pooled = torch.cat([pooled, static_features[:, :self.static_dim]], dim=1)
        return self.success_head(pooled).squeeze(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--test-data-path", type=Path, default=TEST_DATA_PATH)
    parser.add_argument("--processed-data-path", type=Path, default=PROCESSED_DATA_PATH)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--raw-output-csv", type=Path, default=RAW_OUTPUT_CSV)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_tpp_sequences(df, feature_mean, feature_std, static_dim, max_len=MAX_LEN):
    event_seqs = df.select(
        pl.col("journey").list.eval(pl.element().struct.field("ed_id")).alias("event_seq")
    )["event_seq"].to_list()
    delta_seqs = df.select(
        pl.col("journey")
        .list.eval(pl.element().struct.field("event_timestamp").diff().dt.total_seconds().fill_null(0))
        .alias("delta_t_seq")
    )["delta_t_seq"].to_list()
    abs_time_seqs = df.select(
        pl.col("journey")
        .list.eval(
            (pl.element().struct.field("event_timestamp") - pl.element().struct.field("event_timestamp").first()
            ).dt.total_seconds()
        )
        .alias("abs_time_seq")
    )["abs_time_seq"].to_list()

    events_list, deltas_list, masks_list, static_features_list = [], [], [], []
    for ev_seq, dt_seq, abs_seq in zip(event_seqs, delta_seqs, abs_time_seqs):
        length = len(ev_seq)
        e_padded = np.zeros(max_len, dtype=np.int64)
        d_padded = np.zeros(max_len, dtype=np.float32)
        m_padded = np.zeros(max_len, dtype=np.float32)
        log_deltas = np.log1p(np.asarray(dt_seq, dtype=np.float32))
        if length > max_len:
            e_padded[:] = ev_seq[-max_len:]
            d_padded[:] = log_deltas[-max_len:]
            m_padded[:] = 1.0
        else:
            e_padded[:length] = ev_seq
            d_padded[:length] = log_deltas
            m_padded[:length] = 1.0
        events_list.append(e_padded)
        deltas_list.append(d_padded)
        masks_list.append(m_padded)
        static_features_list.append(build_static_features(list(ev_seq), list(dt_seq), list(abs_seq)))

    static_feature_arr = np.array(static_features_list, dtype=np.float32)
    if static_dim > 0:
        if static_feature_arr.shape[1] > len(feature_mean):
            static_feature_arr = static_feature_arr[:, :len(feature_mean)]
        elif static_feature_arr.shape[1] < len(feature_mean):
            padding = np.zeros((len(static_feature_arr), len(feature_mean) - static_feature_arr.shape[1]), dtype=np.float32)
            static_feature_arr = np.concatenate([static_feature_arr, padding], axis=1)
        static_feature_arr = ((static_feature_arr - feature_mean) / feature_std).astype(np.float32)
        static_feature_arr = static_feature_arr[:, :static_dim]
    else:
        static_feature_arr = np.zeros((len(static_features_list), 0), dtype=np.float32)

    return np.array(events_list), np.array(deltas_list), np.array(masks_list), static_feature_arr


def load_model(model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    
    # Infer dimensions from the state dict
    num_events = state_dict["embedding.weight"].shape[0]
    hidden_dim = state_dict["rnn.weight_hh_l0"].shape[0] // 4
    success_input_dim = state_dict["success_head.0.weight"].shape[1]
    static_dim = success_input_dim - (3 * hidden_dim)
    
    model = LegacyNeuralTPP(num_events=num_events, hidden_dim=hidden_dim, static_dim=static_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded: num_events={num_events}, hidden_dim={hidden_dim}, static_dim={static_dim}")
    return model


def main():
    args = parse_args()
    device = get_device()

    print("Loading test data...")
    df_test = pl.read_parquet(args.test_data_path)
    print(f"  {len(df_test)} journeys")

    processed_data = np.load(args.processed_data_path)
    static_feature_mean = processed_data["static_feature_mean"]
    static_feature_std = processed_data["static_feature_std"]

    model = load_model(args.model_path, device)
    static_dim = model.static_dim

    event_arr, delta_arr, mask_arr, static_feature_arr = prepare_tpp_sequences(
        df_test, static_feature_mean, static_feature_std, static_dim
    )

    dataset = TensorDataset(
        torch.tensor(event_arr, dtype=torch.long),
        torch.tensor(delta_arr, dtype=torch.float32),
        torch.tensor(mask_arr, dtype=torch.float32),
        torch.tensor(static_feature_arr, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    probs = []
    with torch.no_grad():
        for events, deltas, masks, static_features in loader:
            events = events.to(device)
            deltas = deltas.to(device)
            masks = masks.to(device)
            static_features = static_features.to(device)
            logits = model.predict_success_logits(events, deltas, masks, static_features)
            probs.append(torch.sigmoid(logits).cpu().numpy())

    probs = np.concatenate(probs)
    tiered_preds = np.zeros_like(probs)
    tiered_preds[(probs >= TIER_1_THRESHOLD) & (probs < TIER_2_THRESHOLD)] = 0.15
    tiered_preds[(probs >= TIER_2_THRESHOLD) & (probs < TIER_3_THRESHOLD)] = 0.5
    tiered_preds[probs >= TIER_3_THRESHOLD] = probs[probs >= TIER_3_THRESHOLD]

    submission = pd.DataFrame({"id": df_test["id"].to_list(), "order_shipped": tiered_preds})
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_csv, index=False)

    raw = pd.DataFrame({"id": df_test["id"].to_list(), "prob": probs, "tiered": tiered_preds})
    args.raw_output_csv.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.raw_output_csv, index=False)

    print(f"\nScored {len(submission):,} rows")
    print(f"Submission: {args.output_csv}")
    print(f"Prob min/mean/max: {probs.min():.6f} / {probs.mean():.6f} / {probs.max():.6f}")
    print(f"Tier 0 (0.0): {(tiered_preds == 0.0).sum():,}")
    print(f"Tier 1 (0.15): {(tiered_preds == 0.15).sum():,}")
    print(f"Tier 2 (0.5): {(tiered_preds == 0.5).sum():,}")
    print(f"Tier 3 (raw): {(probs >= TIER_3_THRESHOLD).sum():,}")


if __name__ == "__main__":
    main()