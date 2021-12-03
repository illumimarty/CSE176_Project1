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

def toDataLoader(subset):
    labels = []
    for i in range(len(subset)):
        if subset[i][1] == 1:
            labels.append(0)
        if subset[i][1] == 2:
            labels.append(1)
        if subset[i][1] == 4:
            labels.append(2)
        if subset[i][1] == 7:
            labels.append(3)
        if subset[i][1] == 8:
            labels.append(4)

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

def main():

    epochs = 100
    min_valid_loss = np.inf

    ######################################################
    digits = [1,2,4,7,8]
    train_data_th = datasets.MNIST(root='./datasets', download=True, train=True, transform=ToTensor())
    test_data_th = datasets.MNIST(root='./datasets', download=True, train=False, transform=ToTensor())

    train_loader, valid_loader = getDataset(digits, train_data_th, split=True)
    test_loader = getDataset(digits, test_data_th, split=False)
    
    print("Initialize the network")
    model = ReferenceNet()
        
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
    # scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=9)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print("Begin training...")
    for e in range(epochs):
        # Train the model and compute the training loss
        train_loss = 0.0
        for data, labels in tqdm(train_loader):
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

        print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_loader)}')

        # Compute the loss on the validation set
        valid_loss = 0.0
        model.eval()     # Optional when not using Model Specific layer
        for data, labels in valid_loader:
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            
            # Forward Pass
            target = model(data)
            # Find the Loss
            loss = criterion(target,labels)
            # Calculate Loss, number of misclassified images by the model on the validation set
            valid_loss += loss.item()

        print(f'\t\t\t Validation Loss: {valid_loss / len(valid_loader)}')
        
        # Check to see if the optimal model can be updated
        min_loss = update_optimal_model(model, min_valid_loss, valid_loss)
        min_valid_loss = min_loss

        curr_lr = optimizer.param_groups[0]['lr']
        print(f'\t\t\t LR: {curr_lr}')

        # if e < 9:
        #     scheduler1.step(valid_loss/len(valid_loader))
        # else:
        scheduler2.step()

if __name__== "__main__":
    main()