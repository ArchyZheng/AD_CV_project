# %%
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
from torchvision import transforms as T
from torchvision.io import read_image

# %%
transform = nn.Sequential(
    T.Resize((224, 224))
)

class TestDataset(Dataset):
    def __init__(self, data_file: str, picture_list_dir: str):
        self.dir_list = [data_file, picture_list_dir]
        self.picture_list = pd.read_csv(self.dir_list[0] + '/' + self.dir_list[1])
        # self.label_list = pd.read_csv(self.dir_list[0] + '/' + self.dir_list[2], sep=" ")
        self.picture_dir_name = self.dir_list[0]

    def __getitem__(self, index):
        img_path = self.picture_dir_name + '/' + str(self.picture_list.values[index][0])
        img_tensor = read_image(img_path).float()
        # the operation of resize must in the dataset, cause the input of different picture has different size
        img_tensor = transform(img_tensor)
        return img_tensor, img_path

    def __len__(self):
        return len(self.picture_list)

#%%
import torch
batch_size = 128
dataset = TestDataset(data_file='./FashionDataset', picture_list_dir='split/test.txt')
test_dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size)
# %%
new_tensor_feature = []
for X, _ in test_dataloader:
    new_tensor_feature.append(X)

new_tensor_feature = torch.cat(new_tensor_feature, dim=0)
# %%
print(new_tensor_feature.shape)
#%%
torch.save(new_tensor_feature, "test_tensor.pt")