import argparse
import polars as pl
import numpy as np
from pathlib import Path

from TPP_features import STATIC_FEATURE_NAMES, SUCCESS_ID, build_static_features, open_prefix

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT / "data" / "truncated_tpp_training.parquet"
FALLBACK_DATA_PATH = ROOT / "data" / "training_new_data.parquet"
DEFAULT_SAVE_PATH = ROOT / "data" / "tpp_processed_data.npz"

MAX_LEN = 128


def parse_args():
    parser = argparse.ArgumentParser(description="Build padded NeuralTPP arrays.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DATA_PATH if DEFAULT_DATA_PATH.exists() else FALLBACK_DATA_PATH,
        help="Input grouped or truncated journey parquet.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_SAVE_PATH)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    return parser.parse_args()


def build_processed_df(df: pl.DataFrame) -> pl.DataFrame:
    if "label_order_shipped" in df.columns:
        label_expr = pl.col("label_order_shipped").cast(pl.Int8)
    else:
        label_expr = (
            pl.col("journey")
            .list.eval(pl.element().struct.field("ed_id") == SUCCESS_ID)
            .list.any()
            .cast(pl.Int8)
            .alias("label_order_shipped")
        )

    optional_exprs = []
    for col_name in ["cutoff_elapsed_seconds", "seconds_since_last_event_at_cutoff"]:
        if col_name in df.columns:
            optional_exprs.append(pl.col(col_name).cast(pl.Float64))
        else:
            optional_exprs.append(pl.lit(None, dtype=pl.Float64).alias(col_name))

    return df.with_columns(label_expr.alias("label_order_shipped")).select(
        [
            pl.col("id"),
            pl.col("label_order_shipped"),
            pl.col("journey").list.eval(pl.element().struct.field("ed_id")).alias("event_seq"),
            pl.col("journey")
            .list.eval(
                pl.element().struct.field("event_timestamp").diff().dt.total_seconds().fill_null(0)
            )
            .alias("delta_t_seq"),
            pl.col("journey")
            .list.eval(
                (
                    pl.element().struct.field("event_timestamp")
                    - pl.element().struct.field("event_timestamp").first()
                ).dt.total_seconds()
            )
            .alias("abs_time_seq"),
            *optional_exprs,
        ]
    )

def prepare_tpp_inputs(df, max_len):
    events_list = []
    deltas_list = []
    masks_list = []
    static_features_list = []
    labels_list = []
    
    # Pre-convert to list for faster iteration
    event_seqs = df["event_seq"].to_list()
    delta_seqs = df["delta_t_seq"].to_list()
    abs_seqs = df["abs_time_seq"].to_list()
    labels = df["label_order_shipped"].to_list()
    cutoff_elapsed_values = df["cutoff_elapsed_seconds"].to_list()
    seconds_since_last_values = df["seconds_since_last_event_at_cutoff"].to_list()
    
    for ev_seq, dt_seq, abs_seq, label, cutoff_elapsed, seconds_since_last in zip(
        event_seqs,
        delta_seqs,
        abs_seqs,
        labels,
        cutoff_elapsed_values,
        seconds_since_last_values,
    ):
        ev_seq = list(ev_seq)
        dt_seq = list(dt_seq)
        abs_seq = list(abs_seq)
        ev_seq, dt_seq, abs_seq = open_prefix(ev_seq, dt_seq, abs_seq, label)
        length = len(ev_seq)
        if length < 2:
            continue
        
        # Initialize zero-filled containers (0 is our Padding ID)
        e_padded = np.zeros(max_len, dtype=np.int64)
        d_padded = np.zeros(max_len, dtype=np.float32)
        m_padded = np.zeros(max_len, dtype=np.float32)
        
        # Apply Log-Scaling to Deltas immediately
        # np.log1p(x) is log(1 + x), essential for handling 0-second gaps
        log_deltas = np.log1p(dt_seq)
        
        if length > max_len:
            # Truncate: Keep MOST RECENT 128 events
            e_padded[:] = ev_seq[-max_len:]
            d_padded[:] = log_deltas[-max_len:]
            m_padded[:] = 1.0
        else:
            # Post-padding: Real data at start, zeros at end
            e_padded[:length] = ev_seq
            d_padded[:length] = log_deltas
            m_padded[:length] = 1.0
            
        events_list.append(e_padded)
        deltas_list.append(d_padded)
        masks_list.append(m_padded)
        static_features_list.append(
            build_static_features(
                ev_seq,
                dt_seq,
                abs_seq,
                cutoff_elapsed_seconds=cutoff_elapsed,
                seconds_since_last_event_at_cutoff=seconds_since_last,
            )
        )
        labels_list.append(label)
        
    static_feature_arr = np.array(static_features_list, dtype=np.float32)
    labels_arr = np.array(labels_list, dtype=np.float32)
    return (
        np.array(events_list),
        np.array(deltas_list),
        np.array(masks_list),
        static_feature_arr,
        labels_arr,
    )

def main():
    args = parse_args()
    df = pl.read_parquet(args.input)
    processed_df = build_processed_df(df)
    processed_df = processed_df.filter(pl.col("event_seq").list.len() >= 2)

    print(f"Input: {args.input}")
    print(f"Processed {len(processed_df)} journeys.")
    print(processed_df.head())

    event_arr, delta_arr, mask_arr, static_feature_arr, labels = prepare_tpp_inputs(
        processed_df, args.max_len
    )
    static_feature_arr_raw = static_feature_arr.astype(np.float32)
    static_feature_mean = static_feature_arr.mean(axis=0).astype(np.float32)
    static_feature_std = static_feature_arr.std(axis=0).astype(np.float32)
    static_feature_std = np.where(static_feature_std < 1e-6, 1.0, static_feature_std).astype(np.float32)
    static_feature_arr = ((static_feature_arr - static_feature_mean) / static_feature_std).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        event_arr=event_arr,
        delta_arr=delta_arr,
        mask_arr=mask_arr,
        static_feature_arr=static_feature_arr,
        static_feature_arr_raw=static_feature_arr_raw,
        static_feature_mean=static_feature_mean,
        static_feature_std=static_feature_std,
        static_feature_names=STATIC_FEATURE_NAMES,
        labels=labels,
    )

    print(f"Arrays saved successfully to {args.output}")


if __name__ == "__main__":
    main()
