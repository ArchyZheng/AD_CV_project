from torch.utils.data import Dataset
import torch
class fashionDataset(Dataset):
    def __init__(self, feature_dir: str):
        super().__init__()
        self.feature = torch.load(feature_dir)

    def __getitem__(self, index):
        return self.feature[index]

    def __len__(self):
        return self.feature.shape[0]