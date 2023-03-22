# %%
import torch
from torch.utils.data import Dataset, DataLoader


# %%
class fashionDataset(Dataset):
    def __init__(self, feature_dir: str, label_dir: str):
        super().__init__()
        self.feature = torch.load(feature_dir)
        self.label = torch.load(label_dir)

    def __getitem__(self, index):
        return self.feature[index], self.label[index]

    def __len__(self):
        return self.feature.shape[0]