from __future__ import annotations

import numpy as np


SUCCESS_ID = 28
SECONDS_PER_HOUR = 60 * 60
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR

APPLICATION_IDS = frozenset({3, 12, 13, 14, 15, 16, 17, 19})
APPROVAL_IDS = frozenset({12, 15})
FIRST_PURCHASE_IDS = frozenset({4, 5, 6, 7, 11, 18})
ORDER_INTENT_IDS = frozenset({5, 6, 7, 11, 18})
DOWNPAYMENT_IDS = frozenset({8, 25, 26, 27})
PROSPECTING_IDS = frozenset({2, 9, 10, 20, 21, 22, 23})

MILESTONE_BY_EVENT = {
    12: 1,
    15: 1,
    7: 2,
    18: 2,
    29: 3,
    8: 4,
    25: 4,
    26: 5,
    27: 5,
}

STATIC_FEATURE_NAMES = np.array(
    [
        "log1p_journey_length",
        "log1p_total_elapsed_seconds",
        "log1p_last_gap_seconds",
        "log1p_mean_gap_seconds",
        "log1p_median_gap_seconds",
        "log1p_max_gap_seconds",
        "zero_gap_share",
        "log1p_unique_event_count",
        "event_diversity_ratio",
        "first_event_id",
        "last_event_id",
        "max_milestone_seen",
        "seen_application",
        "seen_approval",
        "seen_first_purchase",
        "seen_order_intent",
        "seen_downpayment",
        "seen_prospecting",
        "log1p_application_count",
        "log1p_first_purchase_count",
        "log1p_order_intent_count",
        "log1p_downpayment_count",
        "log1p_prospecting_count",
        "order_intent_share",
        "event_velocity_per_day",
        "application_velocity_per_day",
        "order_intent_velocity_per_day",
        "events_last_5h",
        "events_last_1d",
        "events_last_3d",
        "intent_last_5h",
        "intent_last_1d",
        "intent_last_3d",
        "log1p_seconds_since_last_application",
        "log1p_seconds_since_last_order_intent",
        "log1p_seconds_since_last_downpayment",
        "log1p_cutoff_elapsed_seconds",
        "log1p_seconds_since_last_event_at_cutoff",
    ],
    dtype=str,
)


def open_prefix(
    ev_seq: list[int],
    dt_seq: list[float],
    abs_seq: list[float],
    label: bool | int | float,
) -> tuple[list[int], list[float], list[float]]:
    if label and SUCCESS_ID in ev_seq:
        success_idx = ev_seq.index(SUCCESS_ID)
        ev_seq = ev_seq[:success_idx]
        dt_seq = dt_seq[:success_idx]
        abs_seq = abs_seq[:success_idx]
    return ev_seq, dt_seq, abs_seq


def build_static_features(
    ev_seq: list[int],
    dt_seq: list[float],
    abs_seq: list[float],
    cutoff_elapsed_seconds: float | None = None,
    seconds_since_last_event_at_cutoff: float | None = None,
) -> list[float]:
    length = len(ev_seq)
    if length == 0:
        return [0.0] * len(STATIC_FEATURE_NAMES)

    deltas = np.maximum(np.asarray(dt_seq, dtype=np.float32), 0.0)
    abs_times = np.maximum(np.asarray(abs_seq, dtype=np.float32), 0.0)
    total_elapsed = float(abs_times[-1]) if abs_times.size else 0.0
    cutoff_elapsed = (
        max(float(cutoff_elapsed_seconds), 0.0)
        if cutoff_elapsed_seconds is not None and np.isfinite(cutoff_elapsed_seconds)
        else total_elapsed
    )
    seconds_since_last_at_cutoff = (
        max(float(seconds_since_last_event_at_cutoff), 0.0)
        if seconds_since_last_event_at_cutoff is not None
        and np.isfinite(seconds_since_last_event_at_cutoff)
        else 0.0
    )
    elapsed_days = max(total_elapsed / SECONDS_PER_DAY, 1.0 / SECONDS_PER_DAY)
    cutoff_5h = total_elapsed - (5 * SECONDS_PER_HOUR)
    cutoff_1d = total_elapsed - SECONDS_PER_DAY
    cutoff_3d = total_elapsed - (3 * SECONDS_PER_DAY)

    application_count = 0
    approval_count = 0
    first_purchase_count = 0
    order_intent_count = 0
    downpayment_count = 0
    prospecting_count = 0
    events_last_5h = 0
    events_last_1d = 0
    events_last_3d = 0
    intent_last_5h = 0
    intent_last_1d = 0
    intent_last_3d = 0
    last_application_time = None
    last_order_intent_time = None
    last_downpayment_time = None
    max_milestone = 0

    for event, abs_time in zip(ev_seq, abs_times):
        event = int(event)
        abs_time = float(abs_time)
        is_order_intent = event in ORDER_INTENT_IDS

        if event in APPLICATION_IDS:
            application_count += 1
            last_application_time = abs_time
        if event in APPROVAL_IDS:
            approval_count += 1
        if event in FIRST_PURCHASE_IDS:
            first_purchase_count += 1
        if is_order_intent:
            order_intent_count += 1
            last_order_intent_time = abs_time
        if event in DOWNPAYMENT_IDS:
            downpayment_count += 1
            last_downpayment_time = abs_time
        if event in PROSPECTING_IDS:
            prospecting_count += 1

        max_milestone = max(max_milestone, MILESTONE_BY_EVENT.get(event, 0))
        if total_elapsed <= 0.0 or abs_time >= cutoff_5h:
            events_last_5h += 1
            if is_order_intent:
                intent_last_5h += 1
        if total_elapsed <= 0.0 or abs_time >= cutoff_1d:
            events_last_1d += 1
            if is_order_intent:
                intent_last_1d += 1
        if total_elapsed <= 0.0 or abs_time >= cutoff_3d:
            events_last_3d += 1
            if is_order_intent:
                intent_last_3d += 1

    def seconds_since_last(last_time: float | None) -> float:
        if last_time is None:
            return total_elapsed + SECONDS_PER_DAY
        return max(total_elapsed - last_time, 0.0)

    unique_count = len(set(ev_seq))
    mean_gap = float(deltas.mean()) if deltas.size else 0.0
    median_gap = float(np.median(deltas)) if deltas.size else 0.0
    max_gap = float(deltas.max()) if deltas.size else 0.0
    zero_gap_share = float((deltas == 0.0).mean()) if deltas.size else 0.0

    return [
        np.log1p(length),
        np.log1p(total_elapsed),
        np.log1p(float(deltas[-1]) if deltas.size else 0.0),
        np.log1p(mean_gap),
        np.log1p(median_gap),
        np.log1p(max_gap),
        zero_gap_share,
        np.log1p(unique_count),
        unique_count / max(length, 1),
        float(ev_seq[0]),
        float(ev_seq[-1]),
        float(max_milestone),
        float(application_count > 0),
        float(approval_count > 0),
        float(first_purchase_count > 0),
        float(order_intent_count > 0),
        float(downpayment_count > 0),
        float(prospecting_count > 0),
        np.log1p(application_count),
        np.log1p(first_purchase_count),
        np.log1p(order_intent_count),
        np.log1p(downpayment_count),
        np.log1p(prospecting_count),
        order_intent_count / max(length, 1),
        length / elapsed_days,
        application_count / elapsed_days,
        order_intent_count / elapsed_days,
        float(events_last_5h),
        float(events_last_1d),
        float(events_last_3d),
        float(intent_last_5h),
        float(intent_last_1d),
        float(intent_last_3d),
        np.log1p(seconds_since_last(last_application_time)),
        np.log1p(seconds_since_last(last_order_intent_time)),
        np.log1p(seconds_since_last(last_downpayment_time)),
        np.log1p(cutoff_elapsed),
        np.log1p(seconds_since_last_at_cutoff),
    ]
