"""
CP Profile for tpp_model_epoch_3.pth (RNN/LSTM architecture).
Finds a borderline observation (~50% probability) and shows how changing each
static feature shifts the predicted success probability.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_derek" / "TPP"))

from tpp_legacy_predict import LegacyNeuralTPP

MODEL_PATH = ROOT / "kaggle_derek" / "TPP" / "tpp_model_epoch_3.pth"
DATA_PATH = ROOT / "data" / "tpp_processed_data.npz"
OUTPUT_DIR = ROOT / "derek" / "tpp_cp_output"
DEVICE = torch.device("cpu")

# Features to perturb in CP profiles (subset of the 38 static features)
CP_FEATURES = [
    "log1p_journey_length",
    "log1p_total_elapsed_seconds",
    "log1p_last_gap_seconds",
    "log1p_mean_gap_seconds",
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
    "order_intent_share",
    "event_velocity_per_day",
    "intent_last_5h",
    "intent_last_1d",
]


def load_data_and_model():
    data = np.load(DATA_PATH)
    sd = torch.load(MODEL_PATH, map_location=DEVICE)
    num_events = sd["embedding.weight"].shape[0]
    hidden_dim = sd["rnn.weight_hh_l0"].shape[0] // 4
    success_input_dim = sd["success_head.0.weight"].shape[1]
    static_dim = success_input_dim - (3 * hidden_dim)

    model = LegacyNeuralTPP(num_events=num_events, hidden_dim=hidden_dim, static_dim=static_dim).to(DEVICE)
    model.load_state_dict(sd)
    model.eval()
    return model, data


def predict_probs(model, events, deltas, masks, static_features, batch_size=2048):
    dataset = TensorDataset(
        torch.tensor(events, dtype=torch.long),
        torch.tensor(deltas, dtype=torch.float32),
        torch.tensor(masks, dtype=torch.float32),
        torch.tensor(static_features, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs = []
    with torch.no_grad():
        for ev, dt, mk, sf in loader:
            logits = model.predict_success_logits(ev, dt, mk, sf)
            probs.append(torch.sigmoid(logits).numpy())
    return np.concatenate(probs)


def find_borderline_observation(data, model):
    """Find an observation with prediction closest to 0.5."""
    events = data["event_arr"]
    deltas = data["delta_arr"]
    masks = data["mask_arr"]
    static_z = data["static_feature_arr"]

    print("Scoring all observations (this takes a moment)...")
    probs = predict_probs(model, events, deltas, masks, static_z)

    distance = np.abs(probs - 0.5)
    idx = np.argmin(distance)
    print(f"Borderline observation at index {idx}")
    print(f"  Predicted probability: {probs[idx]:.6f}")
    print(f"  Actual label: {int(data['labels'][idx])}")
    print(f"  Distance from 0.5: {distance[idx]:.6f}")

    return idx, probs[idx]


def build_cp_profiles(model, data, ref_idx, static_feature_names_raw, ref_raw):
    """For each static feature, vary it while holding everything else fixed and re-predict."""
    events = data["event_arr"]
    deltas = data["delta_arr"]
    masks = data["mask_arr"]
    static_mean = data["static_feature_mean"]
    static_std = data["static_feature_std"]

    # Get the reference journey (fixed across all perturbations)
    ref_events = events[ref_idx:ref_idx+1]
    ref_deltas = deltas[ref_idx:ref_idx+1]
    ref_masks = masks[ref_idx:ref_idx+1]

    results = []
    for feat_idx, feat_name in enumerate(CP_FEATURES):
        if feat_idx >= len(static_feature_names_raw):
            continue

        ref_val = float(ref_raw[feat_idx])

        # Build a grid of values for this feature (from 1st to 99th percentile)
        feat_col = data["static_feature_arr_raw"][:, feat_idx]
        valid = feat_col[np.isfinite(feat_col)]
        low, high = np.quantile(valid, [0.02, 0.98])
        if np.isclose(low, high):
            low, high = ref_val * 0.5, ref_val * 2.0 if ref_val > 0 else (0, 1)
        grid = np.linspace(low, high, 60)
        # Ensure ref_val is in the grid
        grid = np.unique(np.sort(np.append(grid, ref_val)))

        # For each grid value, create a copy of the reference with this feature changed
        batch_raw = np.tile(ref_raw, (len(grid), 1)).astype(np.float32)
        batch_raw[:, feat_idx] = grid
        # Z-score normalize
        batch_z = ((batch_raw - static_mean) / static_std).astype(np.float32)

        batch_events = np.tile(ref_events, (len(grid), 1))
        batch_deltas = np.tile(ref_deltas, (len(grid), 1))
        batch_masks = np.tile(ref_masks, (len(grid), 1))

        probs = predict_probs(model, batch_events, batch_deltas, batch_masks, batch_z, batch_size=256)

        for val, prob in zip(grid, probs):
            results.append({
                "profile_name": "borderline_tpp",
                "feature": feat_name,
                "feature_value": float(val),
                "predicted_success_probability": float(prob),
                "reference_value": ref_val,
            })

    return pd.DataFrame(results)


def plot_cp_curves(cp_df, ref_row, output_path):
    features = cp_df["feature"].unique()
    n_cols = 4
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4.5 * n_rows), squeeze=False)
    axes = axes.flatten()

    for ax, feature in zip(axes, features):
        part = cp_df[cp_df["feature"] == feature].sort_values("feature_value")
        ref_val = part["reference_value"].iloc[0]
        ref_prob = np.interp(
            ref_val,
            part["feature_value"].to_numpy(),
            part["predicted_success_probability"].to_numpy(),
        )

        ax.plot(part["feature_value"], part["predicted_success_probability"], color="#276FBF", linewidth=2)
        ax.scatter([ref_val], [ref_prob], color="#D64045", s=45, zorder=5)
        ax.axvline(ref_val, color="#D64045", alpha=0.35, linewidth=1)
        ax.set_title(feature, fontsize=10)
        ax.set_xlabel("Feature value")
        ax.set_ylabel("P(order shipped)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)

    for ax in axes[len(features):]:
        ax.axis("off")

    fig.suptitle(
        f"TPP Model CP Profiles — Borderline Observation\n"
        f"User ID: {ref_row.get('user_id', '?')} | "
        f"Base Probability: {ref_row.get('probability', 0.5):.4f}\n"
        f"Actual Outcome: {'Success' if ref_row.get('label', -1) == 1 else 'Failure'}",
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Plot saved to {output_path}")


def print_observation_details(ref_idx, data, static_feature_names, ref_raw, prob, label):
    print("\n" + "="*80)
    print(f"BORDERLINE OBSERVATION DETAILS")
    print("="*80)
    print(f"Index in dataset: {ref_idx}")
    print(f"Predicted probability: {prob:.4f}")
    print(f"Actual label: {'success (1)' if label == 1 else 'failure (0)'}")
    print()

    # Categorical (binary / seen_*) features
    print("--- Categorical/Binary Features ---")
    for i, name in enumerate(static_feature_names):
        val = ref_raw[i]
        if name.startswith("seen_") or name.startswith("has_"):
            print(f"  {name}: {int(val)} ({'yes' if val >= 0.5 else 'no'})")

    # Numeric features (show a subset)
    print("\n--- Numeric Features ---")
    numeric_subset = [
        "log1p_journey_length",
        "log1p_total_elapsed_seconds",
        "log1p_last_gap_seconds",
        "zero_gap_share",
        "log1p_unique_event_count",
        "event_diversity_ratio",
        "first_event_id",
        "last_event_id",
        "max_milestone_seen",
        "order_intent_share",
        "event_velocity_per_day",
        "intent_last_5h",
        "intent_last_1d",
    ]
    for name in numeric_subset:
        if name in static_feature_names:
            i = list(static_feature_names).index(name)
            print(f"  {name}: {ref_raw[i]:.4f}")

    # Journey length
    events = data["event_arr"][ref_idx]
    masks = data["mask_arr"][ref_idx]
    journey_len = int(masks.sum())
    print(f"\n--- Journey Summary ---")
    print(f"  Journey length (events): {journey_len}")
    event_ids = events[:journey_len].tolist() if journey_len > 0 else []
    print(f"  Event sequence: {event_ids[:10]}{'...' if len(event_ids) > 10 else ''}")
    print(f"  First event: {event_ids[0] if event_ids else 'N/A'}, Last event: {event_ids[-1] if event_ids else 'N/A'}")


def main():
    import sys

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model and data...")
    model, data = load_data_and_model()
    static_feature_names = data["static_feature_names"]
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params, {static_feature_names.shape[0]} static features")

    # Find borderline observation
    ref_idx, prob = find_borderline_observation(data, model)
    ref_raw = data["static_feature_arr_raw"][ref_idx]
    label = data["labels"][ref_idx]

    # Print details
    print_observation_details(ref_idx, data, static_feature_names, ref_raw, prob, label)

    # Build CP profiles
    print("\nBuilding CP profiles...")
    cp_df = build_cp_profiles(model, data, ref_idx, static_feature_names, ref_raw)
    cp_df.to_csv(OUTPUT_DIR / "tpp_cp_profile_values.csv", index=False)
    print(f"CP profile values saved to {OUTPUT_DIR / 'tpp_cp_profile_values.csv'}")

    # Reference row for plot title
    ref_row = {
        "user_id": str(ref_idx),
        "probability": prob,
        "label": label,
    }

    # Plot
    output_path = OUTPUT_DIR / "borderline_tpp_cp.png"
    plot_cp_curves(cp_df, ref_row, output_path)

    # Also save summary text
    summary_path = OUTPUT_DIR / "tpp_cp_summary.md"
    lines = [
        "# TPP Model CP Profile — Borderline Observation",
        "",
        f"**Model**: `tpp_model_epoch_3.pth` (RNN/LSTM, hidden_dim=96, embed_dim=32)",
        f"**Dataset index**: {ref_idx}",
        f"**Predicted probability**: {prob:.4f}",
        f"**Actual label**: {'Success' if label == 1 else 'Failure'}",
        "",
        "## Numerical Features",
    ]
    numeric_subset = [
        "log1p_journey_length", "log1p_total_elapsed_seconds",
        "log1p_last_gap_seconds", "zero_gap_share",
        "log1p_unique_event_count", "event_diversity_ratio",
        "first_event_id", "last_event_id", "max_milestone_seen",
        "order_intent_share", "event_velocity_per_day",
        "intent_last_5h", "intent_last_1d",
    ]
    for name in numeric_subset:
        if name in static_feature_names:
            i = list(static_feature_names).index(name)
            lines.append(f"- **{name}**: `{ref_raw[i]:.4f}`")

    lines.append("\n## Categorical (Binary) Features\n")
    for i, name in enumerate(static_feature_names):
        val = ref_raw[i]
        if name.startswith("seen_"):
            lines.append(f"- **{name}**: `{int(val)}`")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()