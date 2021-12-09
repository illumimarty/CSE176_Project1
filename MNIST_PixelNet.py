import numpy as np
from torch import nn
from torchvision import datasets
import torchvision
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
from Neural_Network import process
from torch.utils.tensorboard import SummaryWriter

class PixelNet(nn.Module):
    def __init__(self):
        super(PixelNet,self).__init__()
        # Input layer, transform the image to 100 neurons 
        self.fc1 = nn.Linear(28*28, 100)
        # Hidden layer 1 -> 2, 100 neurons to 50 neurons
        self.fc2 = nn.Linear(100, 50)
        # Hidden layer 2 -> Output layer, 50 neurons to 10 ouput classes
        self.fc3 = nn.Linear(50, 10)
        # Defining the activation function as ReLu (consider using the softmax function, multi-class)
        self.relu = nn.ReLU()

        # nn.CrossEntropyLoss() implements LogSoftMax and NLLloss therefore we do not neet Softmax()
        # self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        # Flattens the image into an object of the following dimensions: (batch_size x 784)
        x = x.view(-1,28*28)
        #print(x.size())
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main(): 

    epochs = 200
    min_valid_loss = np.inf
    version = "PixelNet"
    writer = SummaryWriter("runs/PixelNet")
    ######################################################

    print("Spliting the dataset...")
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())
    train_set, valid_set = random_split(train_set,[50000,10000])
    
    # DataLoaders are iterable data objects
    trainloader = DataLoader(train_set, batch_size=1024)
    validloader = DataLoader(valid_set, batch_size=1024)

    example =  iter(trainloader)
    example_image, example_target = example.next()
    img_grid = torchvision.utils.make_grid(example_image)
    writer.add_image("MNIST Images", img_grid)
    writer.close()
    
    print("Initialize the network")
    model = PixelNet()

    process(version, model, trainloader, validloader, epochs, min_valid_loss)
    
if __name__== "__main__":
    main()