import torch

def organize_output(y_hat, k):
    basic = torch.zeros_like(y_hat)
    _, top_k = y_hat.topk(k, dim=1)
    for i in range(y_hat.shape[0]):
        basic[i, top_k[i, :]] = 1

    return basic