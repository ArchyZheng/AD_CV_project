# %%
from dataset_test import fashionDataset_1
import torch
from model import baseResnet
from torch.utils.data import DataLoader
from utlis import formal_output
import numpy as np
import tqdm


# %%
def test(model_base, model_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = baseResnet(baseModel=model_base)
    test_dataset = fashionDataset_1('test_tensor.pt')
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=100)
    model.load_state_dict(torch.load(f'model-{model_name}.pt', map_location=('cpu')))
    # model.load_state_dict(torch.load(f'model-{model_name}.pt'))
    index_1 = []
    count = 0
    for X in test_dataloader:
        y_hat = model(X)
        index = formal_output(y_hat)
        index_1.append(index)
    index = torch.cat(index_1, dim=0)
    index = index.numpy()
    np.savetxt(f"prediction_{model_name}.txt", index, fmt="%.d")


test(model_base="Resnet50", model_name='9b86h35x')
