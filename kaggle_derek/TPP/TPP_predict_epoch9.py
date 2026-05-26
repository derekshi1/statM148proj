"""
TPP Predictions using tpp_model_epoch_9.pth

Since tpp_model_epoch_9.pth doesn't include a success_head, this script:
1. Uses epoch_9's encoder (events, deltas -> hidden states)
2. Uses epoch_6's success_head (trained classifier on top of hidden states)

Both models have HIDDEN_DIM=64, so their hidden state dimensions are compatible.
This combines the latest encoder (epoch_9) with a trained success classifier (epoch_6).
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

from TPP_class import NeuralTPP
from TPP_features import build_static_features


MAX_LEN = 128
EMBED_DIM = 32
HIDDEN_DIM = 96

ENCODER_MODEL_PATH = SCRIPT_DIR / "tpp_model_epoch_9.pth"
CLASSIFIER_MODEL_PATH = SCRIPT_DIR / "tpp_model_epoch_6.pth"
TEST_DATA_PATH = ROOT / "kaggle_derek" / "open_journeys2_correct.parquet"
PROCESSED_DATA_PATH = ROOT / "data" / "tpp_processed_data.npz"
OUTPUT_CSV = ROOT / "kaggle_derek" / "tpp_submission_epoch9_correct.csv"
RAW_OUTPUT_CSV = ROOT / "kaggle_derek" / "TPP" / "tpp_epoch9_raw_predictions_correct.csv"

TIER_1_THRESHOLD = 0.40
TIER_2_THRESHOLD = 0.60
TIER_3_THRESHOLD = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make predictions using epoch_9 encoder with epoch_6 success_head."
    )
    parser.add_argument("--encoder-model-path", type=Path, default=ENCODER_MODEL_PATH)
    parser.add_argument("--classifier-model-path", type=Path, default=CLASSIFIER_MODEL_PATH)
    parser.add_argument("--test-data-path", type=Path, default=TEST_DATA_PATH)
    parser.add_argument("--processed-data-path", type=Path, default=PROCESSED_DATA_PATH)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--raw-output-csv", type=Path, default=RAW_OUTPUT_CSV)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_tpp_sequences(
    df: pl.DataFrame,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    static_dim: int,
    max_len: int = MAX_LEN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            (
                pl.element().struct.field("event_timestamp")
                - pl.element().struct.field("event_timestamp").first()
            ).dt.total_seconds()
        )
        .alias("abs_time_seq")
    )["abs_time_seq"].to_list()

    events_list = []
    deltas_list = []
    masks_list = []
    static_features_list = []

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
            static_feature_arr = static_feature_arr[:, : len(feature_mean)]
        elif static_feature_arr.shape[1] < len(feature_mean):
            padding = np.zeros(
                (len(static_feature_arr), len(feature_mean) - static_feature_arr.shape[1]),
                dtype=np.float32,
            )
            static_feature_arr = np.concatenate([static_feature_arr, padding], axis=1)
        static_feature_arr = ((static_feature_arr - feature_mean) / feature_std).astype(np.float32)
        static_feature_arr = static_feature_arr[:, :static_dim]
    else:
        static_feature_arr = np.zeros((len(static_features_list), 0), dtype=np.float32)

    return np.array(events_list), np.array(deltas_list), np.array(masks_list), static_feature_arr


def load_encoder_model(model_path: Path, device: torch.device) -> NeuralTPP:
    """Load encoder model (without success_head)."""
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    
    # Rename keys from old RNN naming convention
    key_map = {}
    for key in state_dict.keys():
        if key.startswith("embedding.") and not key.startswith("event_embedding."):
            key_map[key] = key.replace("embedding.", "event_embedding.", 1)
    for old_k, new_k in key_map.items():
        state_dict[new_k] = state_dict.pop(old_k)
    
    emb_key = "event_embedding.weight" if "event_embedding.weight" in state_dict else "embedding.weight"
    num_events = state_dict[emb_key].shape[0]
    
    # Detect hidden_dim from RNN weights if available
    if "rnn.weight_hh_l0" in state_dict:
        rnn_hh = state_dict["rnn.weight_hh_l0"].shape[0]
        hidden_dim = rnn_hh // 4
    else:
        hidden_dim = state_dict[emb_key].shape[1]
    
    # Encoder-only model doesn't have static features
    model = NeuralTPP(num_events=num_events, hidden_dim=hidden_dim, static_dim=0).to(device)
    
    # Load only the encoder weights (skip success_head, RNN-specific, TPP heads, and embedding)
    # Embedding is skipped because legacy checkpoints have different embedding dim (32 vs 64)
    encoder_state = {k: v for k, v in state_dict.items() if not k.startswith("success_head") and not k.startswith("event_linear") and not k.startswith("time_influence") and not k.startswith("intensity_bias") and not k.startswith("rnn.") and "embedding" not in k}
    model.load_state_dict(encoder_state, strict=False)
    model.eval()
    return model


def load_classifier_model(model_path: Path, device: torch.device) -> NeuralTPP:
    """Load full model with success_head classifier."""
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    
    # Rename keys from old RNN naming convention  
    key_map = {}
    for key in state_dict.keys():
        if key.startswith("embedding.") and not key.startswith("event_embedding."):
            key_map[key] = key.replace("embedding.", "event_embedding.", 1)
    for old_k, new_k in key_map.items():
        state_dict[new_k] = state_dict.pop(old_k)
    
    emb_key = "event_embedding.weight" if "event_embedding.weight" in state_dict else "embedding.weight"
    num_events = state_dict[emb_key].shape[0]
    
    # Detect if this is a legacy RNN-based checkpoint and get correct hidden_dim
    # RNN: rnn.weight_hh_l0 has shape (4*hidden_dim, hidden_dim) for LSTM
    if "rnn.weight_hh_l0" in state_dict:
        rnn_hh = state_dict["rnn.weight_hh_l0"].shape[0]
        hidden_dim = rnn_hh // 4  # 4 gates in LSTM
    else:
        hidden_dim = state_dict[emb_key].shape[1]
    
    # Use the actual dimensions from the checkpoint
    success_head_weight_shape = state_dict["success_head.0.weight"].shape
    success_input_dim = success_head_weight_shape[1]
    
    legacy_success_head = success_input_dim < (3 * hidden_dim)
    if legacy_success_head:
        static_dim = max(success_input_dim - hidden_dim, 0)
        model = NeuralTPP(num_events=num_events, hidden_dim=hidden_dim, static_dim=0).to(device)
        # Don't load encoder weights from legacy RNN — shape mismatch with transformer.
        # Only the success_head will be used.
        model.success_head = nn.Sequential(
            nn.Linear(success_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        ).to(device)
        model.success_head.load_state_dict(
            {k.removeprefix("success_head."): v for k, v in state_dict.items() if k.startswith("success_head.")}
        )
    else:
        static_dim = success_input_dim - (3 * hidden_dim)
        model = NeuralTPP(num_events=num_events, hidden_dim=hidden_dim, static_dim=static_dim).to(device)
        model.load_state_dict(state_dict)
    model.eval()

    model.static_dim = static_dim
    model.legacy_success_head = legacy_success_head
    print(
        f"Loading classifier: hidden_dim={hidden_dim}, static_dim={static_dim}, "
        f"success_input_dim={success_input_dim}, legacy_head={legacy_success_head}"
    )
    return model


def predict_success_probs_hybrid(
    encoder_model: NeuralTPP,
    classifier_model: NeuralTPP,
    event_arr: np.ndarray,
    delta_arr: np.ndarray,
    mask_arr: np.ndarray,
    static_feature_arr: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """
    Generate predictions using:
    - epoch_9 encoder for hidden state extraction
    - best model's success_head for classification
    """
    dataset = TensorDataset(
        torch.tensor(event_arr, dtype=torch.long),
        torch.tensor(delta_arr, dtype=torch.float32),
        torch.tensor(mask_arr, dtype=torch.float32),
        torch.tensor(static_feature_arr, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs = []

    with torch.no_grad():
        for events, deltas, masks, static_features in loader:
            events = events.to(device)
            deltas = deltas.to(device)
            masks = masks.to(device)
            static_features = static_features.to(device)

            # Get hidden states from encoder (epoch_9)
            h_encoder = encoder_model.encode(events, deltas)
            
            if getattr(classifier_model, "legacy_success_head", False):
                last_real_idx = masks.sum(dim=1).long().clamp(min=1) - 1
                batch_idx = torch.arange(events.shape[0], device=events.device)
                final_h = h_encoder[batch_idx, last_real_idx]
                if classifier_model.static_dim:
                    final_h = torch.cat([final_h, static_features[:, : classifier_model.static_dim]], dim=1)
                success_logits = classifier_model.success_head(final_h).squeeze(1)
            else:
                success_logits = classifier_model.predict_success_logits_from_hidden(
                    h_encoder, events, masks, static_features
                )
            success_probs = torch.sigmoid(success_logits)
            probs.append(success_probs.cpu().numpy())

    return np.concatenate(probs)


def apply_tiered_predictions(probs: np.ndarray) -> np.ndarray:
    tiered_preds = np.zeros_like(probs)
    tiered_preds[(probs >= TIER_1_THRESHOLD) & (probs < TIER_2_THRESHOLD)] = 0.15
    tiered_preds[(probs >= TIER_2_THRESHOLD) & (probs < TIER_3_THRESHOLD)] = 0.5
    tiered_preds[probs >= TIER_3_THRESHOLD] = probs[probs >= TIER_3_THRESHOLD]
    return tiered_preds


def main() -> None:
    args = parse_args()
    device = get_device()

    print("Loading test data...")
    df_test = pl.read_parquet(args.test_data_path)
    processed_data = np.load(args.processed_data_path)
    print(f"Loading classifier model from {args.classifier_model_path}...")
    classifier_model = load_classifier_model(args.classifier_model_path, device)

    event_arr, delta_arr, mask_arr, static_feature_arr = prepare_tpp_sequences(
        df_test,
        feature_mean=processed_data["static_feature_mean"],
        feature_std=processed_data["static_feature_std"],
        static_dim=classifier_model.static_dim,
        max_len=MAX_LEN,
    )

    print(f"Loading encoder model from {args.encoder_model_path}...")
    encoder_model = load_encoder_model(args.encoder_model_path, device)
    
    print("Generating predictions...")
    probs = predict_success_probs_hybrid(
        encoder_model=encoder_model,
        classifier_model=classifier_model,
        event_arr=event_arr,
        delta_arr=delta_arr,
        mask_arr=mask_arr,
        static_feature_arr=static_feature_arr,
        device=device,
        batch_size=args.batch_size,
    )
    tiered_preds = apply_tiered_predictions(probs)

    submission = pd.DataFrame(
        {
            "id": df_test["id"].to_list(),
            "order_shipped": tiered_preds,
        }
    )
    submission.to_csv(args.output_csv, index=False)

    raw_predictions = pd.DataFrame(
        {
            "id": df_test["id"].to_list(),
            "tpp_success_classifier_prob": probs,
            "tiered_order_shipped": tiered_preds,
        }
    )
    raw_predictions.to_csv(args.raw_output_csv, index=False)

    print(f"\n{'='*80}")
    print(f"Encoder model: {args.encoder_model_path.name}")
    print(f"Classifier model: {args.classifier_model_path.name}")
    print(f"Scored rows: {len(submission):,}")
    print(f"Submission saved to: {args.output_csv}")
    print(f"Raw probabilities saved to: {args.raw_output_csv}")
    print(f"Probability min/mean/max: {probs.min():.6f} / {probs.mean():.6f} / {probs.max():.6f}")
    print(f"\nTier Distribution:")
    print(f"  Tier 0 (0.0):   {(tiered_preds == 0.0).sum():,}")
    print(f"  Tier 1 (0.15):  {(tiered_preds == 0.15).sum():,}")
    print(f"  Tier 2 (0.5):   {(tiered_preds == 0.5).sum():,}")
    print(f"  Tier 3 (raw >= {TIER_3_THRESHOLD}): {(probs >= TIER_3_THRESHOLD).sum():,}")


if __name__ == "__main__":
    main()
