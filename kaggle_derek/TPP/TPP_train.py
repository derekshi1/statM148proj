import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import random
import json
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score

# Import your custom modules
from TPP_class import NeuralTPP
from TPP_dataloader import TPPDataset

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
SPLIT_PATH = SCRIPT_DIR / "tpp_train_val_split.npz"
METADATA_PATH = SCRIPT_DIR / "tpp_training_metadata.json"

SEED = 42
TARGET_TRAIN_SUCCESS_RATE = 0.05
TRAIN_SAMPLE_SIZE = 350_000


def resample_to_target_success_rate(labels, target_success_rate, seed):
    labels = np.asarray(labels)
    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)

    n_pos_target = int(len(neg_idx) * target_success_rate / (1.0 - target_success_rate))
    n_pos_target = min(len(pos_idx), max(1, n_pos_target))
    selected_pos = rng.choice(pos_idx, size=n_pos_target, replace=False)
    selected_idx = np.concatenate([selected_pos, neg_idx])
    rng.shuffle(selected_idx)
    return selected_idx


def train(data_path: Path | None = None):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if data_path is None:
        data_path = ROOT / "data" / "tpp_processed_data.npz"

    data = np.load(data_path)

    # -- Resample & split --
    eligible_idx = resample_to_target_success_rate(
        data["labels"],
        target_success_rate=TARGET_TRAIN_SUCCESS_RATE,
        seed=SEED,
    )
    train_idx, val_idx = train_test_split(
        eligible_idx,
        test_size=0.2,
        random_state=SEED,
        stratify=data["labels"][eligible_idx],
    )
    if len(train_idx) > TRAIN_SAMPLE_SIZE:
        train_idx, _ = train_test_split(
            train_idx,
            train_size=TRAIN_SAMPLE_SIZE,
            random_state=SEED,
            stratify=data["labels"][train_idx],
        )

    # -- Normalise static features on train set --
    static_feature_source = (
        data["static_feature_arr_raw"]
        if "static_feature_arr_raw" in data.files
        else data["static_feature_arr"]
    )
    static_feature_mean = static_feature_source[train_idx].mean(axis=0).astype(np.float32)
    static_feature_std = static_feature_source[train_idx].std(axis=0).astype(np.float32)
    static_feature_std = np.where(static_feature_std < 1e-6, 1.0, static_feature_std).astype(np.float32)
    static_feature_arr = ((static_feature_source - static_feature_mean) / static_feature_std).astype(np.float32)

    # -- Dataset & loaders --
    full_dataset = TPPDataset(
        data['event_arr'],
        data['delta_arr'],
        data['mask_arr'],
        data['labels'],
        static_feature_arr,
    )
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=512, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=512, shuffle=False)

    # -- Model --
    NUM_EVENTS = int(data['event_arr'].max() + 1)
    STATIC_DIM = int(data['static_feature_arr'].shape[1])
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    NHEAD = 8
    MAX_LEN = data['event_arr'].shape[1]
    EPOCHS = 5
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = NeuralTPP(
        num_events=NUM_EVENTS,
        hidden_dim=HIDDEN_DIM,
        static_dim=STATIC_DIM,
        num_layers=NUM_LAYERS,
        nhead=NHEAD,
        max_len=MAX_LEN,
    ).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    train_labels = data['labels'][train_idx].astype(np.float32)
    pos_count = max(float(train_labels.sum()), 1.0)
    neg_count = max(float(len(train_labels) - train_labels.sum()), 1.0)
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=DEVICE)

    # -- Save split & metadata --
    np.savez_compressed(
        SPLIT_PATH,
        train_idx=np.asarray(train_idx, dtype=np.int64),
        val_idx=np.asarray(val_idx, dtype=np.int64),
        static_feature_mean=static_feature_mean,
        static_feature_std=static_feature_std,
    )
    METADATA_PATH.write_text(
        json.dumps({
            "seed": SEED,
            "data_path": str(data_path),
            "train_rows": len(train_idx),
            "val_rows": len(val_idx),
            "train_sample_size": TRAIN_SAMPLE_SIZE,
            "raw_success_rate": float(data["labels"].mean()),
            "target_train_success_rate": TARGET_TRAIN_SUCCESS_RATE,
            "actual_train_success_rate": float(train_labels.mean()),
            "num_events": NUM_EVENTS,
            "static_dim": STATIC_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "nhead": NHEAD,
            "epochs": EPOCHS,
            "arch": "transformer",
            "static_features_fit_on_train_only": True,
        }, indent=2)
    )

    # -- Loss --
    def compute_loss(batch):
        events = batch['events'].to(DEVICE)
        deltas = batch['deltas'].to(DEVICE)
        masks = batch['masks'].to(DEVICE)
        labels_b = batch['label'].to(DEVICE)
        static_features = batch['static_features'].to(DEVICE)

        success_logits = model.predict_success_logits(events, deltas, masks, static_features)
        loss = F.binary_cross_entropy_with_logits(success_logits, labels_b, pos_weight=pos_weight)
        return loss, success_logits, labels_b

    # -- Validation --
    def validate():
        model.eval()
        total_loss = 0.0
        probs = []
        labels_list = []
        with torch.no_grad():
            for batch in val_loader:
                loss, success_logits, batch_labels = compute_loss(batch)
                total_loss += loss.item()
                probs.append(torch.sigmoid(success_logits).cpu().numpy())
                labels_list.append(batch_labels.cpu().numpy())
        probs = np.concatenate(probs)
        labels_arr = np.concatenate(labels_list)
        eps = 1e-6
        clipped_probs = np.clip(probs, eps, 1 - eps)
        val_auc = roc_auc_score(labels_arr, probs) if len(np.unique(labels_arr)) > 1 else float("nan")
        val_logloss = log_loss(labels_arr, clipped_probs)
        return {"loss": total_loss / max(len(val_loader), 1), "auc": val_auc, "logloss": val_logloss}

    # -- Training loop --
    best_auc = -np.inf
    best_logloss = np.inf
    print(
        f"Training on {DEVICE} | rows={len(full_dataset):,} | "
        f"train_rows={len(train_idx):,} | val_rows={len(val_idx):,} | "
        f"static_dim={STATIC_DIM} | hidden_dim={HIDDEN_DIM} | layers={NUM_LAYERS} | heads={NHEAD} | "
        f"train_pos_rate={train_labels.mean():.3%} | pos_weight={pos_weight.item():.3f}",
        flush=True,
    )
    if "static_feature_names" in data.files:
        print("Static features:", ", ".join(data["static_feature_names"].astype(str).tolist()), flush=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss, success_logits, labels_b = compute_loss(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch+1}, batch {batch_idx}: "
                    f"logit_min={success_logits.min().item():.2e}, label_mean={labels_b.mean().item():.2e}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            if (batch_idx + 1) % 500 == 0:
                print(f"  Epoch {epoch+1} batch {batch_idx+1}/{len(train_loader)} | loss {total_loss / (batch_idx + 1):.4f}", flush=True)

        val_metrics = validate()
        scheduler.step(val_metrics["logloss"])
        train_loss = total_loss / len(train_loader)
        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Val AUC: {val_metrics['auc']:.4f} "
            f"Val LogLoss: {val_metrics['logloss']:.4f} LR: {optimizer.param_groups[0]['lr']:.6f}",
            flush=True,
        )
        if (epoch + 1) % 3 == 0:
            torch.save(model.state_dict(), SCRIPT_DIR / f"tpp_model_new_epoch_{epoch+1}.pth")
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(model.state_dict(), SCRIPT_DIR / "tpp_model_new_best_auc.pth")
        if val_metrics["logloss"] < best_logloss:
            best_logloss = val_metrics["logloss"]
            torch.save(model.state_dict(), SCRIPT_DIR / "tpp_model_new_best.pth")

    print(f"Best Val AUC: {best_auc:.4f} | Best Val LogLoss: {best_logloss:.4f}", flush=True)


if __name__ == "__main__":
    train()