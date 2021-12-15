import numpy as np
from clean import *
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from Neural_Network import process
from scipy.io import loadmat

"""layer 1: convolution with 8 kernels of size 3x3 
    layer 2: 2x2 sub-sampling 
    layer 3: convolution with 25 kernels of size 5x5
    layer 4: convolution with 84 kernels of size 4x4 
    layer 5: 2x1 sub-sampling, classification layer"""
       
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5,self).__init__()
        self.fc1 = nn.Linear(500, 5)
        self.fc2 = nn.Linear(5, 5)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=500, kernel_size=5)

        # Max pooling will activate 1the features with the most presence
        # We can, try average pooling -  reflects the average of features (smooths image)
        self.pool = nn.MaxPool2d(kernel_size=2)

        # For a more modern approach use nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.pool(x)
        x = self.tanh(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

def main(): 
    
    epochs = 200
    min_valid_loss = np.inf
    version = "LeNet5"
    ######################################################

    # Images need to be padded to 32x32
    padding = transforms.Compose([transforms.Resize((32, 32)),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5,), (0.5,))])

    print("Spliting the dataset...")
    lenet5 = loadmat('datasets/MNIST-LeNet5.mat')
    # print(lenet5)
    x_train, y_train, x_test, y_test = extractMNISTmini(lenet5, 'train_fea', 'train_gnd', 'test_fea', 'test_gnd')

    # train_set = datasets.MNIST(root='./data', train=True, download=True, transform=padding)
    # test_set = datasets.MNIST(root='./data', train=False, download=True, transform=padding)



    digits = np.array([1,2,4,7,8]) + 1
    x_train, y_train = datasetByDigitList(digits, x_train, y_train)
    print(x_train.shape)
    print(y_train.shape)
    

    # train_set, valid_set = random_split(train_set,[50000,10000])
    # print([train_set])
    # # DataLoaders are iterable data objects
    # trainloader = DataLoader(train_set, batch_size=128)
    # validloader = DataLoader(valid_set, batch_size=128)

    # print("Initialize the network")
    # model = LeNet5()
    # print(train_set[1][1])
    
    # process(version, model, trainloader, validloader, epochs, min_valid_loss)
    
if __name__== "__main__":
    main()