import torch
from dataset import get_train_or_val_dataloader

#%%
batch_size = 128
train_dataloader = get_train_or_val_dataloader(data_file='./FashionDataset', picture_list_dir='split/train.txt',
                                               label_list_dir='split/train_attr.txt', shuffle=True,
                                               batch_size=batch_size)

#%%
new_tensor_feature = []
new_label = []
for X, y in train_dataloader:
    new_tensor_feature.append(X)
    new_label.append(y)

new_tensor_feature = torch.cat(new_tensor_feature, dim=0)
new_label = torch.cat(new_label, dim=0)
#%%
torch.save(new_tensor_feature, "train_tensor.pt")
#%%
torch.save(new_label, "train_label.pt")
#%%
val_dataloader = get_train_or_val_dataloader(data_file='./FashionDataset', picture_list_dir='split/val.txt',
                                             label_list_dir='split/val_attr.txt', shuffle=True,
                                             batch_size=batch_size)
#%%
new_tensor_feature = []
new_label = []
for X, y in val_dataloader:
    new_tensor_feature.append(X)
    new_label.append(y)

new_tensor_feature = torch.cat(new_tensor_feature, dim=0)
new_label = torch.cat(new_label, dim=0)
#%%
#%%
torch.save(new_tensor_feature, "val_tensor.pt")
#%%
torch.save(new_label, "val_label.pt")