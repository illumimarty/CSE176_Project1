import torch
#import visdom
import numpy as np
from torch import nn
from tqdm import tqdm
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split

"""layer 1: convolution with 8 kernels of size 3x3 
    layer 2: 2x2 sub-sampling 
    layer 3: convolution with 25 kernels of size 5x5
    layer 4: convolution with 84 kernels of size 4x4 
    layer 5: 2x1 sub-sampling, classification layer"""
       
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5,self).__init__()
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)

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
        x = self.tanh(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
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

def process(model, trainloader, validloader, epochs, min_valid_loss):
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

    iterations = 0 # for visdom, currently unused
    #min_valid_loss = np.inf

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

def main(): 
    
    epochs = 5
    iterations = 0  # for visdom, currently unused
    min_valid_loss = np.inf

    ######################################################

    # Images need to be padded to 32x32
    padding = transforms.Compose([transforms.Resize((32, 32)),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5,), (0.5,))])

    print("Spliting the dataset...")
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=padding)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=padding)
    train_set, valid_set = random_split(train_set,[50000,10000])
    
    # DataLoaders are iterable data objects
    trainloader = DataLoader(train_set, batch_size=128)
    validloader = DataLoader(valid_set, batch_size=128)

    print("Initialize the network")
    model = LeNet5()
    
    process(model, trainloader, validloader, epochs, min_valid_loss)
    
if __name__== "__main__":
    main()