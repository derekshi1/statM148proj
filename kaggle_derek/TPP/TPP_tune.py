"""
TPP Hyperparameter Tuning
Optimize tiering thresholds and other hyperparameters using a balanced validation set
that matches the class imbalance of the testing data (95% no order shipped, 5% order shipped).

Note: This script requires a model with a success_head (tpp_model_best.pth or tpp_model_epoch_6.pth).
The tpp_model_epoch_9.pth does not include the success_head, so use --model-path to specify
a different checkpoint if needed.
"""

import argparse
import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from TPP_class import NeuralTPP
from TPP_dataloader import TPPDataset

# Constants
MAX_LEN = 128
EMBED_DIM = 32
HIDDEN_DIM = 96

MODEL_PATH = SCRIPT_DIR / "tpp_model_best.pth"
PROCESSED_DATA_PATH = ROOT / "data" / "tpp_processed_data.npz"
SPLIT_PATH = SCRIPT_DIR / "tpp_train_val_split.npz"

# Optional: specify epoch_9 model with --model-path flag
# Note: epoch_9 model may not have success_head; use tpp_model_best.pth or tpp_model_epoch_6.pth

# Test set class imbalance (95% negative, 5% positive)
TARGET_POS_RATIO = 0.05
TARGET_NEG_RATIO = 0.95


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_path: Path, device: torch.device) -> NeuralTPP:
    """Load a trained TPP model from checkpoint."""
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)

    # Infer dimensions from the transformer event_embedding and success_head
    num_events, hidden_dim = state_dict["event_embedding.weight"].shape
    success_input_dim = state_dict["success_head.0.weight"].shape[1]
    static_dim = success_input_dim - (3 * hidden_dim)
    if static_dim < 0:
        static_dim = 0

    model = NeuralTPP(
        num_events=num_events,
        hidden_dim=hidden_dim,
        static_dim=static_dim,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def normalize_static_features(processed_data: np.lib.npyio.NpzFile, split_data: np.lib.npyio.NpzFile | None) -> np.ndarray:
    static_source = (
        processed_data["static_feature_arr_raw"]
        if "static_feature_arr_raw" in processed_data.files
        else processed_data["static_feature_arr"]
    )
    if split_data is not None:
        mean = split_data["static_feature_mean"]
        std = split_data["static_feature_std"]
    else:
        mean = processed_data["static_feature_mean"]
        std = processed_data["static_feature_std"]
    return ((static_source - mean) / std).astype(np.float32)


def align_static_features(static_feature_arr: np.ndarray, static_dim: int) -> np.ndarray:
    """Match validation features to the loaded checkpoint's static input width."""
    current_dim = static_feature_arr.shape[1]
    if current_dim == static_dim:
        return static_feature_arr
    if current_dim > static_dim:
        return static_feature_arr[:, :static_dim]
    padding = np.zeros((len(static_feature_arr), static_dim - current_dim), dtype=np.float32)
    return np.concatenate([static_feature_arr, padding], axis=1)


def create_balanced_validation_set(
    labels: np.ndarray,
    event_arr: np.ndarray,
    delta_arr: np.ndarray,
    mask_arr: np.ndarray,
    static_feature_arr: np.ndarray,
    pos_ratio: float = TARGET_POS_RATIO,
    val_size: int = 50000,
    random_state: int = 42,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a balanced validation set matching the target class imbalance.
    Returns validation indices and the validation dataset.
    """
    np.random.seed(random_state)
    
    # Split into positive and negative samples
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]
    
    # Calculate required counts
    num_pos = int(val_size * pos_ratio)
    num_neg = val_size - num_pos
    
    # Ensure we have enough samples
    num_pos = min(num_pos, len(pos_indices))
    num_neg = min(num_neg, len(neg_indices))
    
    # Randomly sample
    selected_pos = np.random.choice(pos_indices, size=num_pos, replace=False)
    selected_neg = np.random.choice(neg_indices, size=num_neg, replace=False)
    val_indices = np.concatenate([selected_pos, selected_neg])
    
    val_labels = labels[val_indices]
    val_event_arr = event_arr[val_indices]
    val_delta_arr = delta_arr[val_indices]
    val_mask_arr = mask_arr[val_indices]
    val_static_feature_arr = static_feature_arr[val_indices] if static_feature_arr is not None else None
    
    actual_pos_ratio = val_labels.mean()
    print(f"Validation set: {len(val_indices):,} samples, "
          f"positive ratio: {actual_pos_ratio:.4f} ({actual_pos_ratio*100:.1f}%)")
    
    return val_indices, val_labels, val_event_arr, val_delta_arr, val_mask_arr, val_static_feature_arr


def load_validation_set(
    labels: np.ndarray,
    event_arr: np.ndarray,
    delta_arr: np.ndarray,
    mask_arr: np.ndarray,
    static_feature_arr: np.ndarray,
    split_path: Path,
    val_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reuse the model's held-out validation split when available."""
    if split_path.exists():
        split_data = np.load(split_path)
        val_indices = split_data["val_idx"]
        if len(val_indices) > val_size:
            rng = np.random.default_rng(seed)
            val_labels_full = labels[val_indices]
            pos_indices = val_indices[val_labels_full == 1]
            neg_indices = val_indices[val_labels_full == 0]
            num_pos = min(int(val_size * TARGET_POS_RATIO), len(pos_indices))
            num_neg = min(val_size - num_pos, len(neg_indices))
            selected_pos = rng.choice(pos_indices, size=num_pos, replace=False)
            selected_neg = rng.choice(neg_indices, size=num_neg, replace=False)
            val_indices = np.concatenate([selected_pos, selected_neg])

        val_labels = labels[val_indices]
        print(
            f"Validation set from saved split: {len(val_indices):,} samples, "
            f"positive ratio: {val_labels.mean():.4f} ({val_labels.mean()*100:.1f}%)"
        )
        return (
            val_indices,
            val_labels,
            event_arr[val_indices],
            delta_arr[val_indices],
            mask_arr[val_indices],
            static_feature_arr[val_indices],
        )

    print("Saved train/validation split not found; falling back to sampled validation set.")
    return create_balanced_validation_set(
        labels,
        event_arr,
        delta_arr,
        mask_arr,
        static_feature_arr,
        pos_ratio=TARGET_POS_RATIO,
        val_size=val_size,
        random_state=seed,
    )


def predict_success_probs(
    model: NeuralTPP,
    event_arr: np.ndarray,
    delta_arr: np.ndarray,
    mask_arr: np.ndarray,
    static_feature_arr: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Generate success probability predictions for validation set."""
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


def apply_tiered_predictions(
    probs: np.ndarray,
    tier1_threshold: float,
    tier2_threshold: float,
    tier3_threshold: float,
) -> np.ndarray:
    """Apply tiered predictions based on thresholds."""
    tiered_preds = np.zeros_like(probs)
    tiered_preds[(probs >= tier1_threshold) & (probs < tier2_threshold)] = 0.15
    tiered_preds[(probs >= tier2_threshold) & (probs < tier3_threshold)] = 0.5
    tiered_preds[probs >= tier3_threshold] = probs[probs >= tier3_threshold]
    return tiered_preds


def evaluate_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    tier1_threshold: float,
    tier2_threshold: float,
    tier3_threshold: float,
) -> dict:
    """Evaluate tiering thresholds using multiple metrics."""
    tiered_preds = apply_tiered_predictions(probs, tier1_threshold, tier2_threshold, tier3_threshold)
    
    # Log loss. Raw probabilities measure calibration; tiered log loss is the
    # objective for the actual Kaggle submission values.
    eps = 1e-6
    clipped_probs = np.clip(probs, eps, 1 - eps)
    clipped_tiered = np.clip(tiered_preds, eps, 1 - eps)
    raw_logloss = log_loss(labels, clipped_probs)
    tiered_logloss = log_loss(labels, clipped_tiered)
    
    # ROC AUC
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else np.nan
    
    # Distribution metrics
    tier_0 = (tiered_preds == 0.0).sum()
    tier_1 = (tiered_preds == 0.15).sum()
    tier_2 = (tiered_preds == 0.5).sum()
    tier_3 = (probs >= tier3_threshold).sum()
    
    return {
        "tier1_threshold": tier1_threshold,
        "tier2_threshold": tier2_threshold,
        "tier3_threshold": tier3_threshold,
        "logloss": tiered_logloss,
        "tiered_logloss": tiered_logloss,
        "raw_logloss": raw_logloss,
        "auc": auc,
        "tier_0_count": tier_0,
        "tier_1_count": tier_1,
        "tier_2_count": tier_2,
        "tier_3_count": tier_3,
        "pred_mean": tiered_preds.mean(),
        "pred_std": tiered_preds.std(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune TPP hyperparameters on a balanced validation set."
    )
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--processed-data-path", type=Path, default=PROCESSED_DATA_PATH)
    parser.add_argument("--split-path", type=Path, default=SPLIT_PATH)
    parser.add_argument("--val-size", type=int, default=50000, help="Validation set size")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--tier1-range", nargs=2, type=float, default=[0.40, 0.75], 
                        help="Tier 1 threshold search range")
    parser.add_argument("--tier2-range", nargs=2, type=float, default=[0.60, 0.85], 
                        help="Tier 2 threshold search range")
    parser.add_argument("--tier3-range", nargs=2, type=float, default=[0.75, 0.95], 
                        help="Tier 3 threshold search range")
    parser.add_argument("--n-steps", type=int, default=5, 
                        help="Number of steps to search in each range")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading data...")
    processed_data = np.load(args.processed_data_path)
    split_data = np.load(args.split_path) if args.split_path.exists() else None
    labels = processed_data["labels"]
    event_arr = processed_data["event_arr"]
    delta_arr = processed_data["delta_arr"]
    mask_arr = processed_data["mask_arr"]
    static_feature_arr = normalize_static_features(processed_data, split_data)
    
    print(f"Loading validation set (target pos ratio: {TARGET_POS_RATIO*100:.1f}% if subsampled)...")
    val_indices, val_labels, val_event_arr, val_delta_arr, val_mask_arr, val_static_feature_arr = load_validation_set(
        labels,
        event_arr,
        delta_arr,
        mask_arr,
        static_feature_arr,
        split_path=args.split_path,
        val_size=args.val_size,
        seed=args.seed,
    )
    
    print(f"\nLoading model from {args.model_path}...")
    model = load_model(args.model_path, device)
    val_static_feature_arr = align_static_features(val_static_feature_arr, model.static_dim)
    
    # Generate predictions
    print("Generating predictions on validation set...")
    probs = predict_success_probs(
        model,
        val_event_arr,
        val_delta_arr,
        val_mask_arr,
        val_static_feature_arr,
        device,
        batch_size=args.batch_size,
    )
    print(f"Probability stats - min: {probs.min():.6f}, mean: {probs.mean():.6f}, max: {probs.max():.6f}\n")
    
    # Generate threshold search space
    tier1_range = np.linspace(args.tier1_range[0], args.tier1_range[1], args.n_steps)
    tier2_range = np.linspace(args.tier2_range[0], args.tier2_range[1], args.n_steps)
    tier3_range = np.linspace(args.tier3_range[0], args.tier3_range[1], args.n_steps)
    
    print(f"Searching {args.n_steps}³ = {args.n_steps**3} threshold combinations...\n")
    
    # Grid search
    results = []
    total_combinations = args.n_steps ** 3
    
    for i, (t1, t2, t3) in enumerate(product(tier1_range, tier2_range, tier3_range)):
        # Skip invalid combinations
        if not (t1 < t2 < t3):
            continue
        
        metrics = evaluate_thresholds(probs, val_labels, t1, t2, t3)
        results.append(metrics)
        
        if (i + 1) % max(1, total_combinations // 10) == 0:
            print(f"Progress: {i + 1}/{total_combinations} combinations evaluated")
    
    # Sort by tiered log loss (ascending - lower is better for submitted values)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("logloss").reset_index(drop=True)
    
    print("\n" + "="*100)
    print("TOP 10 THRESHOLD CONFIGURATIONS (by log loss)")
    print("="*100)
    print(results_df.head(10).to_string(index=False))
    
    # Get best result
    best = results_df.iloc[0]
    print("\n" + "="*100)
    print("BEST CONFIGURATION")
    print("="*100)
    print(f"Tier 1 Threshold: {best['tier1_threshold']:.4f}")
    print(f"Tier 2 Threshold: {best['tier2_threshold']:.4f}")
    print(f"Tier 3 Threshold: {best['tier3_threshold']:.4f}")
    print(f"Log Loss: {best['logloss']:.6f}")
    print(f"ROC AUC: {best['auc']:.6f}")
    print(f"\nTier Distribution:")
    print(f"  Tier 0 (0.0):   {best['tier_0_count']:,} samples")
    print(f"  Tier 1 (0.15):  {best['tier_1_count']:,} samples")
    print(f"  Tier 2 (0.5):   {best['tier_2_count']:,} samples")
    print(f"  Tier 3 (raw):   {best['tier_3_count']:,} samples")
    print(f"  Mean prediction: {best['pred_mean']:.6f}")
    print(f"  Std prediction:  {best['pred_std']:.6f}")
    
    # Save results
    output_csv = SCRIPT_DIR / "tpp_threshold_tuning_results.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\nAll {len(results_df)} results saved to: {output_csv}")
    
    # Save best config
    best_config = {
        "tier1_threshold": float(best['tier1_threshold']),
        "tier2_threshold": float(best['tier2_threshold']),
        "tier3_threshold": float(best['tier3_threshold']),
        "logloss": float(best['logloss']),
        "auc": float(best['auc']),
    }
    
    import json
    config_path = SCRIPT_DIR / "best_tier_config.json"
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"Best config saved to: {config_path}")


if __name__ == "__main__":
    main()
