import torch
from torch import nn, optim
# from torch._C import T
from torch.utils.data import TensorDataset, DataLoader, Subset
from torchvision import datasets,transforms
import numpy as np
from torchvision.transforms.transforms import ToTensor

digits = [1,2,4,7,8]
digit_index = []
labels = []
subset_idx = []
train_data_th = datasets.MNIST(root='./datasets', download=True, train=True, transform=ToTensor())
test_data_th = datasets.MNIST(root='./datasets', download=True, train=False)

# for i in digits:
idx_1 = 1*(train_data_th.targets == 1).nonzero().flatten().tolist()
idx_2 = 1*(train_data_th.targets == 2).nonzero().flatten().tolist()
idx_4 = 1*(train_data_th.targets == 4).nonzero().flatten().tolist()
idx_7 = 1*(train_data_th.targets == 7).nonzero().flatten().tolist()
idx_8 = 1*(train_data_th.targets == 8).nonzero().flatten().tolist()

subset_idx = idx_1+idx_2+idx_4+idx_7+idx_8
# print(len(subset_idx))

train_data_sub = Subset(train_data_th, subset_idx)      # pytorch dataset
for i in range(len(train_data_sub)):
    labels.append(train_data_sub[i][1])

labels = torch.tensor(labels)

train_loader = DataLoader(train_data_sub, batch_size=len(train_data_sub))
train_data = next(iter(train_loader))[0].numpy().reshape([-1,28*28]).astype(np.float32)
train_data_norm = (train_data / 255)
train_data = TensorDataset(torch.from_numpy(train_data_norm), labels)
train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)

# uncomment below to see structure
# for i in range(10):
#     print(train_data[i])


