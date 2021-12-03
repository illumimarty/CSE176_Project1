import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split, TensorDataset

#DIGITS 1,2,4,7,8

class ReferenceNet(nn.Module):
    def __init__(self):
        super(ReferenceNet,self).__init__()
        # Input layer, transform the image to 500 neurons 
        self.fc1 = nn.Linear(28*28, 500)
        # Hidden layer 1 -> 2, 500 neurons to 300 neurons
        self.fc2 = nn.Linear(500, 300)
        # Hidden layer 2 -> Output layer, 300 neurons to 5 ouput classes
        self.fc3 = nn.Linear(300, 5)

        # Defining the activation function as ReLu (consider using the softmax function, multi-class)
        self.relu = nn.ReLU()

    def forward(self, x):
        print("here")
        # Flattens the image into an object of the following dimensions: (batch_size x 784)
        x = x.view(-1,28*28)
        #print(x.size())
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main(): 

    print("Spliting the dataset...")
    train_data = datasets.MNIST(root='./datasets', download=True, train=True, transform=ToTensor())
    test_data = datasets.MNIST(root='./datasets', download=True, train=False)

    data_train = np.array(train_data.data[:]).reshape([-1, 28 * 28]).astype(np.float32)
    data_train = (data_train / 255)
    dtrain_mean = data_train.mean(axis=0)
    train_data.data = data_train - dtrain_mean

    digits = [1, 2, 4, 7, 8]
    subset = []
    for i in range(len(train_data)):
        if train_data.targets[i] in digits:
            subset.append(i)

    train_data.data = torch.from_numpy(train_data.data)
    #print(len(train_data.targets))
    sub_data = torch.utils.data.Subset(train_data, subset)
    sub_targets = torch.utils.data.Subset(train_data.targets, subset) 
    modded_data = TensorDataset(sub_data, sub_targets)
    print(type(modded_data))
    #test_data = torch.utils.data.Subset(data_test, subset)

    train_loader = DataLoader(modded_data, num_workers=4, batch_size=2048, shuffle=True)
    # epoch = 5
    # for e in range(epoch):
    #     for data, labels in tqdm(train_loader):
    #         #print(data[0])
    #         print(labels[0])
    #         break




    #train_data = torch.utils.data.TensorDataset(torch.from_numpy(data_train), train_data_th.targets)
    #test_data = torch.utils.data.TensorDataset(torch.from_numpy(data_test), test_data_th.targets)   
    
if __name__== "__main__":
    main()