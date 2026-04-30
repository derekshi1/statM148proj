import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

class TPPDataset(Dataset):
    def __init__(self, event_arr, delta_arr, mask_arr, labels):
        self.event_arr = torch.tensor(event_arr, dtype=torch.long)
        self.delta_arr = torch.tensor(delta_arr, dtype=torch.float32)
        self.mask_arr = torch.tensor(mask_arr, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'events': self.event_arr[idx],
            'deltas': self.delta_arr[idx],
            'masks': self.mask_arr[idx],
            'label': self.labels[idx]
        }
