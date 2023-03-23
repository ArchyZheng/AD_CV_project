#%%
from dataset_test import fashionDataset
import torch
from model import baseResnet
from torch.utils.data import DataLoader
from utlis import formal_output
import numpy as np
#%%
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = baseResnet().to(device)
    test_dataset = fashionDataset('test_tensor.pt')
    print(len(test_dataset))
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=100)
    model.load_state_dict(torch.load('model.pt', map_location=('cpu')))
    f = open('output.txt', 'a')
    index_1 = []
    for X in test_dataloader:
        y_hat = model(X)
        index = formal_output(y_hat)
        index_1.append(index)
    index = torch.cat(index_1, dim=0)
    index = index.numpy()
    np.savetxt("output.txt", index, fmt="%.d")
#%%
main()
#%%
