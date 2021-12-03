import torch
from torch import nn, optim
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader, Subset, SubsetRandomSampler, random_split
from torch.utils.data.dataset import ConcatDataset
from torchvision import datasets,transforms
import numpy as np
from torchvision.transforms.transforms import ToTensor

from clean import combineDigitData

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

        train_subset, valid_subset = random_split(data_subset, [train_len, val_len], generator=torch.Generator().manual_seed(1))

        train_loader = toDataLoader(train_subset)
        valid_loader = toDataLoader(valid_subset)

        return train_loader, valid_loader

    else:
        data_loader = toDataLoader(data_subset)
        return data_loader


def train_epoch(model,device,dataloader,loss_fn,optimizer):
    train_loss,train_correct=0.0,0
    model.train()
    for images, labels in dataloader:

        images,labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        train_correct += (predictions == labels).sum().item()

    return train_loss,train_correct

def valid_epoch(model,device,dataloader,loss_fn):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    for images, labels in dataloader:

        images,labels = images.to(device),labels.to(device)
        output = model(images)
        loss=loss_fn(output,labels)
        valid_loss+=loss.item()*images.size(0)
        scores, predictions = torch.max(output.data,1)
        val_correct+=(predictions == labels).sum().item()

    return valid_loss,val_correct

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

def main():
    print("Getting the data...")
    digits = [1,2,4,7,8]
    train_data_th = datasets.MNIST(root='./datasets', download=True, train=True, transform=ToTensor())
    test_data_th = datasets.MNIST(root='./datasets', download=True, train=False, transform=ToTensor())

    train_loader, valid_loader = getDataset(digits, train_data_th, split=True)
    # test_loader = getDataset(digits, test_data_th, split=False)

    torch.manual_seed(42)
    criterion = nn.CrossEntropyLoss()

    num_epochs=10
    batch_size=1024
    k=10
    splits=KFold(n_splits=k,shuffle=True,random_state=42)
    foldperf={}

    dataset = ConcatDataset([train_data_th, test_data_th])

    print("Cross validation...")
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

        print('Fold {}'.format(fold + 1))


        # train_sampler = SubsetRandomSampler(train_idx)
        # test_sampler = SubsetRandomSampler(val_idx)
        # train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        # test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
        
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        model = ReferenceNet()
        # model.to(device)
        # optimizer = optim.Adam(model.parameters(), lr=0.002)
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        optimizer = optim.SGD(params, lr=0.01, momentum=0.9, nesterov=True)

        history = {'train_loss': [], 'valid_loss': [],'train_acc':[],'valid_acc':[]}

        for epoch in range(num_epochs):
            train_loss, train_correct=train_epoch(model,device,train_loader,criterion,optimizer)
            valid_loss, valid_correct=valid_epoch(model,device,valid_loader,criterion)

            train_loss = train_loss / len(train_loader)
            train_acc = train_correct / len(train_loader) * 100
            valid_loss = valid_loss / len(valid_loader)
            valid_acc = valid_correct / len(valid_loader) * 100

            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Valid Loss:{:.3f} AVG Training Acc {:.2f} % AVG Valid Acc {:.2f} %".format(epoch + 1,
                                                                                                                num_epochs,
                                                                                                                train_loss,
                                                                                                                valid_loss,
                                                                                                                train_acc,
                                                                                                                valid_acc))
            history['train_loss'].append(train_loss)
            history['valid_loss'].append(valid_loss)
            history['train_acc'].append(train_acc)
            history['valid_acc'].append(valid_acc)

        foldperf['fold{}'.format(fold+1)] = history  

    torch.save(model,'k_cross_CNN.pt')   

if __name__== "__main__":
    main()
