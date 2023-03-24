#%%
from dataset_test import fashionDataset_1
import torch
from model import baseResnet
from torch.utils.data import DataLoader
from utlis import formal_output
import numpy as np
import tqdm
#%%
def test(config, model_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = baseResnet(baseModel=config.baseModel)
    test_dataset = fashionDataset_1('test_tensor.pt')
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=100)
    # model.load_state_dict(torch.load('model.pt', map_location=('cpu')))
    model.load_state_dict(torch.load('model.pt'))
    index_1 = []
    count = 0
    for X in test_dataloader:
        y_hat = model(X)
        index = formal_output(y_hat)
        index_1.append(index)
    index = torch.cat(index_1, dim=0)
    index = index.numpy()
    np.savetxt("prediction.txt", index, fmt="%.d")