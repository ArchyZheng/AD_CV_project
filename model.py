#%%
import torch
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
#%%
class baseResnet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        resnet_1 = resnet50(7)
        self.common_layers = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2
        ) # I want to create a common part of the model
        self.category_1 = nn.Sequential(
            resnet_1.layer3,
            resnet_1.layer4,
            resnet_1.avgpool,
        )
        self.category_1_output = resnet_1.fc
    def forward(self, X):
        common = self.common_layers(X)
        y0 = self.category_1(common)
        y0 = torch.flatten(y0, 1)
        y0 = self.category_1_output(y0)

        return y0