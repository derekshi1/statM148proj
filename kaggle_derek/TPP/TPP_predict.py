from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from TPP_class import NeuralTPP
from TPP_features import build_static_features


MAX_LEN = 128   
EMBED_DIM = 32
HIDDEN_DIM = 96

MODEL_PATH = SCRIPT_DIR / "tpp_model_epoch_3.pth"
TEST_DATA_PATH = ROOT / "kaggle_derek" / "open_journeys2_correct.parquet"
PROCESSED_DATA_PATH = ROOT / "data" / "tpp_processed_data.npz"
SPLIT_PATH = SCRIPT_DIR / "tpp_train_val_split.npz"
OUTPUT_CSV = ROOT / "kaggle_derek" / "tpp_epoch_3_part2_correct.csv"
RAW_OUTPUT_CSV = ROOT / "kaggle_derek" / "TPP" / "tpp_best_raw_predictions_correcty.csv"

TIER_1_THRESHOLD = 0.40
TIER_2_THRESHOLD = 0.60
TIER_3_THRESHOLD = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make Kaggle predictions from the epoch 6 NeuralTPP checkpoint."
    )
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--test-data-path", type=Path, default=TEST_DATA_PATH)
    parser.add_argument("--processed-data-path", type=Path, default=PROCESSED_DATA_PATH)
    parser.add_argument("--split-path", type=Path, default=SPLIT_PATH)
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
    static_dim: int | None = None,
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
    if static_feature_arr.shape[1] > len(feature_mean):
        static_feature_arr = static_feature_arr[:, : len(feature_mean)]
    elif static_feature_arr.shape[1] < len(feature_mean):
        padding = np.zeros(
            (len(static_feature_arr), len(feature_mean) - static_feature_arr.shape[1]),
            dtype=np.float32,
        )
        static_feature_arr = np.concatenate([static_feature_arr, padding], axis=1)
    static_feature_arr = ((static_feature_arr - feature_mean) / feature_std).astype(np.float32)
    if static_dim is not None:
        static_feature_arr = static_feature_arr[:, :static_dim]

    return np.array(events_list), np.array(deltas_list), np.array(masks_list), static_feature_arr


def load_model(model_path: Path, device: torch.device) -> NeuralTPP:
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)

    # Infer dimensions from the embedding and success_head weights
    # Check which key name the checkpoint uses (embedding vs event_embedding)
    emb_key = "event_embedding.weight" if "event_embedding.weight" in state_dict else "embedding.weight"
    num_events, hidden_dim = state_dict[emb_key].shape
    success_input_dim = state_dict["success_head.0.weight"].shape[1]
    static_dim = success_input_dim - (3 * hidden_dim)
    if static_dim < 0:
        static_dim = 0

    # Rename keys if checkpoint was saved with different naming convention
    # (e.g. "embedding" instead of "event_embedding")
    key_map = {}
    for key in state_dict.keys():
        if key.startswith("embedding.") and not key.startswith("event_embedding."):
            new_key = key.replace("embedding.", "event_embedding.", 1)
            key_map[key] = new_key
    for old_k, new_k in key_map.items():
        state_dict[new_k] = state_dict.pop(old_k)

    model = NeuralTPP(
        num_events=num_events,
        hidden_dim=hidden_dim,
        static_dim=static_dim,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_static_normalization(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if args.split_path.exists():
        split_data = np.load(args.split_path)
        return split_data["static_feature_mean"], split_data["static_feature_std"]

    processed_data = np.load(args.processed_data_path)
    return processed_data["static_feature_mean"], processed_data["static_feature_std"]


def predict_success_probs(
    model: NeuralTPP,
    event_arr: np.ndarray,
    delta_arr: np.ndarray,
    mask_arr: np.ndarray,
    static_feature_arr: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
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

            success_logits = model.predict_success_logits(events, deltas, masks, static_features)
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

    df_test = pl.read_parquet(args.test_data_path)
    static_feature_mean, static_feature_std = load_static_normalization(args)
    model = load_model(args.model_path, device)
    event_arr, delta_arr, mask_arr, static_feature_arr = prepare_tpp_sequences(
        df_test,
        feature_mean=static_feature_mean,
        feature_std=static_feature_std,
        static_dim=model.static_dim,
        max_len=MAX_LEN,
    )

    probs = predict_success_probs(
        model=model,
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
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_csv, index=False)

    raw_predictions = pd.DataFrame(
        {
            "id": df_test["id"].to_list(),
            "tpp_success_classifier_prob": probs,
            "tiered_order_shipped": tiered_preds,
        }
    )
    args.raw_output_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_predictions.to_csv(args.raw_output_csv, index=False)

    print(f"Loaded model: {args.model_path}")
    print(f"Scored rows: {len(submission):,}")
    print(f"Submission saved to: {args.output_csv}")
    print(f"Raw probabilities saved to: {args.raw_output_csv}")
    print(f"Probability min/mean/max: {probs.min():.6f} / {probs.mean():.6f} / {probs.max():.6f}")
    print(f"Total Tier 0 (0.0): {(tiered_preds == 0.0).sum():,}")
    print(f"Total Tier 1 (0.15): {(tiered_preds == 0.15).sum():,}")
    print(f"Total Tier 2 (0.5): {(tiered_preds == 0.5).sum():,}")
    print(f"Total Tier 3 (raw >= {TIER_3_THRESHOLD}): {(probs >= TIER_3_THRESHOLD).sum():,}")


if __name__ == "__main__":
    main()
