import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import WeightedRandomSampler, DataLoader, Subset
from sklearn.model_selection import train_test_split

# Import your custom modules
from TPP_class import NeuralTPP, tpp_nll_loss
from TPP_dataloader import TPPDataset

# 1. Load the processed data
DATA_PATH = "/Users/derek/UCLA_stats/statM148proj/data/tpp_processed_data.npz"
data = np.load(DATA_PATH)

# 2. Initialize Dataset
full_dataset = TPPDataset(
    data['event_arr'], 
    data['delta_arr'], 
    data['mask_arr'], 
    data['labels']
)

# 3. Split & Dataloaders
train_idx, val_idx = train_test_split(
    range(len(full_dataset)), 
    test_size=0.2, 
    random_state=42, 
    stratify=data['labels']
)

train_labels = data['labels'][train_idx]
class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
# weight = 1 / count
weight = 1. / class_sample_count
samples_weight = np.array([weight[int(t)] for t in train_labels])

sampler = WeightedRandomSampler(
    weights=torch.from_numpy(samples_weight), 
    num_samples=len(samples_weight), 
    replacement=True
)

# Use this sampler in your train_loader
train_loader = DataLoader(
    Subset(full_dataset, train_idx), 
    batch_size=128, 
    sampler=sampler # DO NOT use shuffle=True with a sampler
)
# 4. Model Setup
NUM_EVENTS = int(data['event_arr'].max() + 1)
EMBED_DIM = 32
HIDDEN_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

model = NeuralTPP(NUM_EVENTS, EMBED_DIM, HIDDEN_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)

def summarize_batch(events, deltas, masks, alpha=None, v=None, event_logits=None):
    real_mask = masks.bool()
    real_deltas = deltas[real_mask]
    summary = {
        "valid_steps": int(real_mask.sum().item()),
        "delta_min": float(real_deltas.min().item()) if real_deltas.numel() else 0.0,
        "delta_max": float(real_deltas.max().item()) if real_deltas.numel() else 0.0,
        "delta_mean": float(real_deltas.mean().item()) if real_deltas.numel() else 0.0,
        "event_min": int(events[real_mask].min().item()) if real_mask.any() else 0,
        "event_max": int(events[real_mask].max().item()) if real_mask.any() else 0,
    }
    if alpha is not None:
        summary["alpha_min"] = float(alpha.min().item())
        summary["alpha_max"] = float(alpha.max().item())
    if v is not None:
        summary["v_min"] = float(v.min().item())
        summary["v_max"] = float(v.max().item())
    if event_logits is not None:
        summary["logit_min"] = float(event_logits.min().item())
        summary["logit_max"] = float(event_logits.max().item())
    return summary

# 5. Training Loop
def train():
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            events = batch['events'].to(DEVICE)
            deltas = batch['deltas'].to(DEVICE)
            masks = batch['masks'].to(DEVICE)
            
            # Shifted Targets
            input_ev, input_dt = events[:, :-1], deltas[:, :-1]
            target_ev, target_dt, target_m = events[:, 1:], deltas[:, 1:], masks[:, 1:]
            
            optimizer.zero_grad()
            alpha, v, event_logits = model(input_ev, input_dt)
            
            loss = tpp_nll_loss(alpha, v, event_logits, target_ev, target_dt, target_m, DEVICE, NUM_EVENTS)
            if not torch.isfinite(loss):
                debug_info = summarize_batch(target_ev, target_dt, target_m, alpha, v, event_logits)
                raise RuntimeError(f"Non-finite loss at epoch {epoch+1}, batch {batch_idx}: {debug_info}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")
    
    # Save the final model
    torch.save(model.state_dict(), "tpp_model.pth")

if __name__ == "__main__":
    train()
