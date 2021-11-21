import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        # Input layer
        self.fc1 = nn.Linear(28*28, 256)
        # Hidden layer 1 -> 2
        self.fc2 = nn.Linear(256, 128)
        # Hidden layer 2 -> Output layer
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(1,-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main(): 
    convert_to_tensor = transforms.Compose([ transforms.ToTensor() ])
    
    train = datasets.MNIST('', train = True, transform = convert_to_tensor, download = True)
    # 50,000 datapoints in the training set, 10,000 datapoints in the validation set
    train, valid = random_split(train,[50000,10000])

    trainloader = DataLoader(train, batch_size=32)
    validloader = DataLoader(valid, batch_size=32)  
    
    net = Network()
    
    if torch.cuda.is_available():
        net = net.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)

    epochs = 5

    for e in range(epochs):
        train_loss = 0.0
        for data, labels in tqdm(trainloader):
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            
            # Clear the gradients
            optimizer.zero_grad()
            # Forward Pass
            target = net(data)
            # Find the Loss
            loss = criterion(target,labels)
            # Calculate gradients 
            loss.backward()
            # Update Weights
            optimizer.step()
            # Calculate Loss
            train_loss += loss.item()
        
        print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(trainloader)}')

if __name__== "__main__":
    main()