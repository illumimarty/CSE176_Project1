import torch
#import visdom
import numpy as np
from torch import nn
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split

class Network(nn.Module):
    """THIS IS NOT LeNet5"""
    def __init__(self):
        super(Network,self).__init__()
        # Input layer, transform the image to 100 neurons 
        self.fc1 = nn.Linear(28*28, 100)
        # Hidden layer 1 -> 2, 100 neurons to 50 neurons
        self.fc2 = nn.Linear(100, 50)
        # Hidden layer 2 -> Output layer, 50 neurons to 10 ouput classes
        self.fc3 = nn.Linear(50, 10)
        # Defining the activation as ReLu
        self.relu = nn.ReLU()

    def forward(self, images):
        # Flattens the image into an object of the following dimensions: (batch_size x 784)
        x = images.view(-1,28*28)
        #print(x.size())
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_nn(model, criterion, optimizer, trainloader):
    train_loss = 0.0
    for data, labels in tqdm(trainloader):
        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        
        # Clear the gradients
        optimizer.zero_grad()
        #Forward Pass
        target = model(data)
        # Find the Loss
        loss = criterion(target,labels)
        # Calculate gradients
        loss.backward()
        # Update Weights
        optimizer.step()
        # Calculate Loss, number of misclassified images by the model
        train_loss += loss.item()
    return train_loss

def validate_nn(model, criterion, validloader):
    valid_loss = 0.0
    model.eval()     # Optional when not using Model Specific layer
    for data, labels in validloader:
        # Transfer Data to GPU if available
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        
        # Forward Pass
        target = model(data)
        # Find the Loss
        loss = criterion(target,labels)
        # Calculate Loss, number of misclassified images by the model on the validation set
        valid_loss += loss.item()
    return valid_loss

def update_optimal_model(model, min_valid_loss, valid_loss):
    if min_valid_loss > valid_loss:
        print(f'\t\t\t Observed Errors Decreased({min_valid_loss:.3f}--->{valid_loss:.3f}) \n\t\t\t Saving The Model')
        min_valid_loss = valid_loss
        
        # Saving State Dict
        torch.save(model.state_dict(), 'saved_model.pth')
        return min_valid_loss
    else:
        print(f'\t\t\t No decrease in observed errors')
        return min_valid_loss

def main(): 
    print("Spliting the dataset...")
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())
    train_set, valid_set = random_split(train_set,[50000,10000])
    
    # DataLoaders are iterable data objects
    trainloader = DataLoader(train_set, batch_size=128)
    validloader = DataLoader(valid_set, batch_size=128)

    print("Initialize the network")
    model = Network()
    
    print("Checking GPU availability...")
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model loaded onto GPU")
    else:
        print("     GPU Not available")
    
    print("Initialize criterion")
    criterion = nn.CrossEntropyLoss()
    print("Initialize optimizer")
    # lr = Learning Rate
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

    epochs = 5
    iterations = 0
    min_valid_loss = np.inf

    print("Begin training...")
    for e in range(epochs):
        # Train the model and compute the training loss
        train_loss = train_nn(model, criterion, optimizer, trainloader)
        print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(trainloader)}')

        # Compute the loss on the validation set
        valid_loss = validate_nn(model, criterion, validloader)
        print(f'\t\t\t Validation Loss: {valid_loss / len(validloader)}')
        
        # Check to see if the optimal model can be updated
        min_loss = update_optimal_model(model, min_valid_loss, valid_loss)
        min_valid_loss = min_loss

if __name__== "__main__":
    main()