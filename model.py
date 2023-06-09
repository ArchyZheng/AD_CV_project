#%%
import torch
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
from copy import deepcopy
#%%
class baseResnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.output_layer = nn.Linear(1000, 26)

    def forward(self, X):
        output = self.resnet(X)
        output = self.output_layer(output)

        return output