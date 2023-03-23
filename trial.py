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