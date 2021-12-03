import torch
from torch import nn, optim
# from torch._C import T
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.utils.data.dataset import random_split
from torchvision import datasets,transforms
import numpy as np
from torchvision.transforms.transforms import ToTensor

from clean import combineDigitData

def toDataLoader(subset):
    labels = []
    for i in range(len(subset)):
        labels.append(subset[i][1])

    labels = torch.tensor(labels)

    # Normalizing data
    data_loader = DataLoader(subset, batch_size=len(subset))
    numpy_data = next(iter(data_loader))[0].numpy().reshape([-1,28*28]).astype(np.float32)
    data_norm = (numpy_data / 255).astype(np.float32)
    data_mean = data_norm.mean(axis=0)
    data_norm -= data_mean
    combined_data = TensorDataset(torch.from_numpy(data_norm), labels)
    data_loader = DataLoader(combined_data, batch_size=len(combined_data), shuffle=True)

    return data_loader

def getDataset(digits, data, split=False):
    digit_list = []
    subset_range = []

    # Getting features of the specified digits from MNIST
    for digit in digits:
        digit_list.append(1*(data.targets == digit).nonzero().flatten().tolist())

    for digit in digit_list:
        subset_range += digit

    # Getting labels
    data_subset = Subset(data, subset_range)

    if split is True:
        train_len = int(0.8*len(data_subset))
        val_len = len(data_subset) - train_len

        train_subset, val_subset = random_split(data_subset, [train_len, val_len], generator=torch.Generator().manual_seed(1))

        train_loader = toDataLoader(train_subset)
        val_loader = toDataLoader(val_subset)

        return train_loader, val_loader

    else:
        data_loader = toDataLoader(data_subset)
        return data_loader

    # for i in range(len(data_subset)):
    #     labels.append(data_subset[i][1])

    # labels = torch.tensor(labels)

    # # Normalizing data
    # data_loader = DataLoader(data_subset, batch_size=len(data_subset))
    # numpy_data = next(iter(data_loader))[0].numpy().reshape([-1,28*28]).astype(np.float32)
    # data_norm = (numpy_data / 255).astype(np.float32)
    # data_mean = data_norm.mean(axis=0)
    # data_norm -= data_mean
    # combined_data = TensorDataset(torch.from_numpy(data_norm), labels)
    # data_loader = DataLoader(combined_data, batch_size=len(combined_data), shuffle=True)

    # return data_loader


def main():
    digits = [1,2,4,7,8]
    train_data_th = datasets.MNIST(root='./datasets', download=True, train=True, transform=ToTensor())
    test_data_th = datasets.MNIST(root='./datasets', download=True, train=False, transform=ToTensor())

    train_loader = getDataset(digits, train_data_th, split=True)
    test_loader = getDataset(digits, test_data_th, split=False)

if __name__== "__main__":
    main()
