# %%
import torch.nn as nn
import torch

# %%
# Example of target with class indices
loss = nn.CrossEntropyLoss()
input = torch.zeros(size=(2, 26))
input[0, [i for i in range(6)]] = 0.5
input[1, [i for i in range(6)]] = 0.5
input = input.softmax(dim=1)
target = torch.zeros(size=(2, 26)).float()
target[0, [i for i in range(6)]] = 1
target[1, [i for i in range(6)]] = 1
target = target.softmax(dim=1)
# %%
target
# %%
output = loss(input, target)
# %%
output
# %%
output.backward()
# %%
# Example of target with class probabilities
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
output = loss(input, target)
output.backward()
#%%
loss = nn.BCEWithLogitsLoss()

input = torch.zeros(size=(1, 26))
input[0, [i for i in range(1, 7)]] = 1
input = input.softmax(dim=1)
target = torch.zeros(size=(1, 26)).float()
target[0, [i for i in range(6)]] = 1
output = loss(input, target)
#%%
print(output)
#%%
import numpy as np
import wandb
from dataset import get_train_or_val_dataloader
wandb.init()
#%%
dataloader = get_train_or_val_dataloader(data_file='./FashionDataset', picture_list_dir='split/train.txt',
                    label_list_dir='split/train_attr.txt', shuffle=True, batch_size=8)
#%%
examples = []
for X, y in dataloader:
 print(X.shape)
 X.transpose_(1, 2)
 X.transpose_(2, 3)
 print(X.shape)
 print(X[0].shape)
 for i in range(3):
  temp = X[i].numpy()
  image = wandb.Image(temp, caption=f"this is just a trial{i}")
  examples.append(image)
 wandb.log({'examples': examples})
 break
#%%


for i in range(3):
 pixels = np.random.randint(low=0, high=256, size=(224, 224, 3))
 image = wandb.Image(pixels, caption=f"random field {i}")
 examples.append(image)
wandb.log({"examples": examples})
#%%
x = torch.randn(2, 3)
print(x)
y = torch.transpose(x, 0, 1)
print(y)