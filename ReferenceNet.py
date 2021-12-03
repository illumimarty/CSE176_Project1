import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split, TensorDataset, Subset

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
    optimizer = torch.optim.SGD(model.parameters(), lr = 1)
    scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=9)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print("Begin training...")
    for e in range(epochs):
        # Train the model and compute the training loss
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

        print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(trainloader)}')

        # Compute the loss on the validation set
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

        print(f'\t\t\t Validation Loss: {valid_loss / len(validloader)}')
        
        # Check to see if the optimal model can be updated
        min_loss = update_optimal_model(model, min_valid_loss, valid_loss)
        min_valid_loss = min_loss

        curr_lr = optimizer.param_groups[0]['lr']
        print(f'\t\t\t LR: {curr_lr}')

        if e < 9:
            scheduler1.step(valid_loss/len(validloader))
        else:
            scheduler2.step(valid_loss/len(validloader))

def main(): 

    epochs = 50
    min_valid_loss = np.inf

    ######################################################

    print("Spliting the dataset...")
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

    print("Initialize the network")
    model = ReferenceNet()
        
    process(model, train_loader, valid_loader, epochs, min_valid_loss)

if __name__== "__main__":
    main()