import argparse
import hashlib
import math
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SUCCESS_ID = 28
INACTIVITY_DAYS = 60
EPSILON = pd.Timedelta(microseconds=1)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build one truncated training snapshot per journey using the logic from "
            "wk2lec.md, then store the lecture's length-based oversampling factor as a "
            "weight column so the output stays tractable."
        )
    )
    parser.add_argument(
        "--input",
        default="training_new_data.parquet",
        help=(
            "Grouped journey parquet. This repo already has training_new_data.parquet, "
            "which is the cleaned event data organized into per-journey lists."
        ),
    )
    parser.add_argument(
        "--event-defs",
        default="Event Definitions.csv",
        help="CSV with event metadata.",
    )
    parser.add_argument(
        "--output",
        default="truncated_data_codex.parquet",
        help="Output parquet path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global seed for deterministic cutoff sampling.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=20000,
        help="How many snapshots to buffer before writing a parquet batch.",
    )
    return parser.parse_args()


def stable_seed(journey_id: str, seed: int) -> int:
    digest = hashlib.blake2b(
        f"{journey_id}|{seed}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little")


def load_event_metadata(path: Path):
    event_defs = pd.read_csv(path)
    event_defs = event_defs.rename(
        columns={
            "event_definition_id": "ed_id",
            "event_name": "event_name_def",
            "stage": "stage_def",
            "milestone_number": "milestone_number_def",
        }
    )
    event_defs["milestone_number_def"] = pd.to_numeric(
        event_defs["milestone_number_def"], errors="coerce"
    )
    event_ids = sorted(event_defs["ed_id"].astype(int).unique().tolist())
    name_map = dict(zip(event_defs["ed_id"], event_defs["event_name_def"]))
    stage_map = dict(zip(event_defs["ed_id"], event_defs["stage_def"]))
    milestone_map = dict(zip(event_defs["ed_id"], event_defs["milestone_number_def"]))
    return event_ids, name_map, stage_map, milestone_map


def first_success_time(journey):
    for event in journey:
        if int(event["ed_id"]) == SUCCESS_ID:
            return event["event_timestamp"]
    return None


def choose_cutoff(start_time, cutoff_upper, journey_id: str, seed: int):
    if cutoff_upper <= start_time:
        return start_time

    total_seconds = max((cutoff_upper - start_time).total_seconds(), 0.0)
    rng = random.Random(stable_seed(journey_id, seed))
    offset_seconds = rng.random() * total_seconds
    return start_time + pd.to_timedelta(offset_seconds, unit="s")


def snapshot_record(
    journey_id: str,
    journey,
    cutoff_time,
    terminal_time,
    label_order_shipped: int,
    terminal_reason: str,
    weight_days: int,
    name_map,
    stage_map,
    milestone_map,
    event_ids,
):
    observed = [event for event in journey if event["event_timestamp"] <= cutoff_time]
    if not observed:
        return None

    cutoff_time = pd.Timestamp(cutoff_time)
    terminal_time = pd.Timestamp(terminal_time)
    first_event = observed[0]
    last_event = observed[-1]
    counts = Counter(int(event["ed_id"]) for event in observed)

    milestones = [
        milestone_map.get(int(event["ed_id"]))
        for event in observed
        if pd.notna(milestone_map.get(int(event["ed_id"])))
    ]
    observed_duration_days = max(
        (cutoff_time - first_event["event_timestamp"]).total_seconds() / 86400.0, 0.0
    )
    days_since_last_event = max(
        (cutoff_time - last_event["event_timestamp"]).total_seconds() / 86400.0, 0.0
    )

    record = {
        "id": journey_id,
        "cutoff_time": cutoff_time,
        "journey_terminal_time": terminal_time,
        "journey_terminal_reason": terminal_reason,
        "label_order_shipped": label_order_shipped,
        "sampling_weight_days": weight_days,
        "n_events_observed": len(observed),
        "n_unique_event_types_observed": len(counts),
        "observed_duration_days": observed_duration_days,
        "days_since_last_event": days_since_last_event,
        "cutoff_hour_utc": cutoff_time.hour,
        "cutoff_dayofweek_utc": cutoff_time.dayofweek,
        "cutoff_month_utc": cutoff_time.month,
        "cutoff_dayofyear_utc": cutoff_time.dayofyear,
        "first_ed_id": int(first_event["ed_id"]),
        "first_event_name": name_map.get(int(first_event["ed_id"])),
        "first_stage": stage_map.get(int(first_event["ed_id"])),
        "last_ed_id": int(last_event["ed_id"]),
        "last_event_name": name_map.get(int(last_event["ed_id"])),
        "last_stage": stage_map.get(int(last_event["ed_id"])),
        "max_milestone_seen": max(milestones) if milestones else None,
    }

    for ed_id in event_ids:
        count = counts.get(ed_id, 0)
        record[f"count_ed_{ed_id}"] = count
        record[f"seen_ed_{ed_id}"] = count > 0

    return record


def flush_records(writer, records):
    if not records:
        return writer

    table = pa.Table.from_pandas(pd.DataFrame.from_records(records), preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(
            where=OUTPUT_PATH,
            schema=table.schema,
            compression="zstd",
        )
    writer.write_table(table)
    records.clear()
    return writer


def main():
    global OUTPUT_PATH

    args = parse_args()
    input_path = Path(args.input)
    event_defs_path = Path(args.event_defs)
    OUTPUT_PATH = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Could not find input parquet: {input_path}")
    if not event_defs_path.exists():
        raise FileNotFoundError(f"Could not find event definitions CSV: {event_defs_path}")

    event_ids, name_map, stage_map, milestone_map = load_event_metadata(event_defs_path)
    parquet_file = pq.ParquetFile(input_path)

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    writer = None
    buffered_records = []
    total_journeys = 0
    total_written = 0
    skipped_empty = 0
    success_count = 0

    for row_group_idx in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group_idx, columns=["id", "journey"])
        batch_df = table.to_pandas()

        for journey_id, journey in zip(batch_df["id"], batch_df["journey"]):
            total_journeys += 1
            if len(journey) == 0:
                skipped_empty += 1
                continue

            start_time = journey[0]["event_timestamp"]
            last_event_time = journey[-1]["event_timestamp"]
            success_time = first_success_time(journey)

            label_order_shipped = int(success_time is not None)
            if label_order_shipped:
                success_count += 1
                terminal_time = success_time
                terminal_reason = "success"
                cutoff_upper = success_time - EPSILON
            else:
                terminal_time = last_event_time + pd.Timedelta(days=INACTIVITY_DAYS)
                terminal_reason = "inactive_60d"
                cutoff_upper = terminal_time - EPSILON

            total_duration_days = max(
                (terminal_time - start_time).total_seconds() / 86400.0, 0.0
            )
            weight_days = max(1, math.ceil(total_duration_days))
            cutoff_time = choose_cutoff(start_time, cutoff_upper, str(journey_id), args.seed)

            record = snapshot_record(
                journey_id=str(journey_id),
                journey=journey,
                cutoff_time=cutoff_time,
                terminal_time=terminal_time,
                label_order_shipped=label_order_shipped,
                terminal_reason=terminal_reason,
                weight_days=weight_days,
                name_map=name_map,
                stage_map=stage_map,
                milestone_map=milestone_map,
                event_ids=event_ids,
            )
            if record is None:
                skipped_empty += 1
                continue

            buffered_records.append(record)
            total_written += 1

            if len(buffered_records) >= args.flush_every:
                writer = flush_records(writer, buffered_records)

        print(
            f"Processed row group {row_group_idx + 1}/{parquet_file.num_row_groups} | "
            f"journeys={total_journeys:,} snapshots={total_written:,}"
        )

    writer = flush_records(writer, buffered_records)
    if writer is not None:
        writer.close()

    print()
    print(f"Input journeys processed: {total_journeys:,}")
    print(f"Snapshots written: {total_written:,}")
    print(f"Skipped empty journeys: {skipped_empty:,}")
    print(f"Success journeys: {success_count:,}")
    print(f"Success rate: {success_count / max(total_written, 1):.4f}")
    print(f"Saved parquet to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
