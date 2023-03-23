# %%
import torch


def organize_output(y_hat, k):
    basic = torch.zeros_like(y_hat)
    _, top_k = y_hat.topk(k, dim=1)
    for i in range(y_hat.shape[0]):
        basic[i, top_k[i, :]] = 1

    return basic


def formal_output(y_hat):
    output = torch.split(y_hat, [7, 3, 3, 4, 6, 3], dim=1)
    index = []
    for i in range(6):
        _, index_1 = output[i].max(1)
        index_1.unsqueeze_(0)
        index.append(index_1)
    index = torch.cat(index, 0)
    index.transpose_(0, 1)
    return index

#%%
import torch

a = torch.randn(size=(64, 26))
index = formal_output(a)
index
#%%
import numpy as np
index = index.numpy()
index = index.astype(int)
#%%
index.dtype
#%%
np.savetxt('nihao.txt', index, fmt='%.d')
#%%
#%%
a = torch.Tensor([1, 2, 3, 4]).numpy()
#%%
a.astype(int)
