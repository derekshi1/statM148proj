import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

class TPPDataset(Dataset):
    def __init__(self, event_arr, delta_arr, mask_arr, labels, static_feature_arr=None):
        self.event_arr = torch.as_tensor(event_arr, dtype=torch.long)
        self.delta_arr = torch.as_tensor(delta_arr, dtype=torch.float32)
        self.mask_arr = torch.as_tensor(mask_arr, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.static_feature_arr = (
            torch.as_tensor(static_feature_arr, dtype=torch.float32)
            if static_feature_arr is not None
            else None
        )
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {
            'events': self.event_arr[idx],
            'deltas': self.delta_arr[idx],
            'masks': self.mask_arr[idx],
            'label': self.labels[idx]
        }
        if self.static_feature_arr is not None:
            item['static_features'] = self.static_feature_arr[idx]
        return item
