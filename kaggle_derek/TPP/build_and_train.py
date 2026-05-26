"""
End-to-end pipeline: training_2.parquet → truncated snapshots → npz → train transformer.

Faster than row-by-row: use polars for the heavy lifting and keep truncated snapshots
in a parquet file that preprocessing_TPP.py can consume.
"""

import sys
from pathlib import Path
import hashlib
import random

import numpy as np
import polars as pl

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from TPP_features import SUCCESS_ID

INPUT_PATH = ROOT / "data" / "training_2.parquet"
SNAPSHOT_PATH = ROOT / "data" / "truncated_tpp_training.parquet"
SEED = 42


def stable_seed(journey_id: str, seed: int) -> int:
    digest = hashlib.blake2b(
        f"{journey_id}|{seed}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little")


def first_success_time(journey):
    for event in journey:
        if int(event["ed_id"]) == SUCCESS_ID:
            return event["event_timestamp"]
    return None


def choose_prefix_length(max_length: int, journey_id: str, seed: int) -> int:
    if max_length <= 1:
        return max_length
    rng = random.Random(stable_seed(journey_id, seed))
    return rng.randint(1, max_length)


def build_snapshot(row, seed: int):
    journey = row["journey"]
    if not journey:
        return None

    journey = sorted(journey, key=lambda e: (e["event_timestamp"], e["ed_id"]))
    success_time = first_success_time(journey)
    label_order_shipped = int(success_time is not None)

    if label_order_shipped:
        available = [
            event
            for event in journey
            if event["event_timestamp"] < success_time and int(event["ed_id"]) != SUCCESS_ID
        ]
    else:
        available = [event for event in journey if int(event["ed_id"]) != SUCCESS_ID]

    if len(available) < 2:
        return None

    prefix_length = choose_prefix_length(len(available), str(row["id"]), seed)
    observed = available[:prefix_length]
    if len(observed) < 2:
        observed = available[:2]

    first_ts = observed[0]["event_timestamp"]
    last_ts = observed[-1]["event_timestamp"]
    cutoff_elapsed = max((last_ts - first_ts).total_seconds(), 0.0)

    return {
        "id": str(row["id"]),
        "label_order_shipped": label_order_shipped,
        "cutoff_elapsed_seconds": cutoff_elapsed,
        "seconds_since_last_event_at_cutoff": 0.0,
        "observed_duration_seconds": cutoff_elapsed,
        "n_events_observed": len(observed),
        "journey": observed,
    }


def main():
    print(f"Loading grouped journeys from {INPUT_PATH}...")
    df = pl.read_parquet(INPUT_PATH, columns=["id", "journey"])
    print(f"  {len(df):,} journeys loaded")

    records = []
    skipped = 0
    count = 0
    for row in df.iter_rows(named=True):
        snap = build_snapshot(row, SEED)
        if snap is None:
            skipped += 1
            continue
        records.append(snap)
        count += 1
        if count % 200_000 == 0:
            print(f"  Processed {count:,} snapshots...", flush=True)

    out = pl.DataFrame(records)
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(SNAPSHOT_PATH, compression="zstd")
    print(f"\nWrote {len(out):,} truncated snapshots to {SNAPSHOT_PATH}")
    print(f"Skipped {skipped:,} journeys with insufficient history")
    print(f"Success rate: {out['label_order_shipped'].mean():.2%}")

    # Now run preprocessing_TPP and training
    print("\n--- Step 2: Build TPP npz arrays ---")
    from preprocessing_TPP import main as preprocess_main
    preprocess_main()

    print("\n--- Step 3: Train transformer ---")
    from TPP_train import train as run_train
    run_train()


if __name__ == "__main__":
    main()