import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import random
from datetime import datetime
from clean import *
from plots import *
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, validation_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main(): 
    """USER INPUT"""
    ## 0 - 5-fold cross-validation and adjusting hyperparameters
    ## 1 - Determining avg testing accuracy
    case = 0
    cm = False # if true and case=1, see confusion matrix

    ## EDIT THESE 2 VARS TO CHANGE DIGIT CLASSES ##
    digit1 = 4
    digit2 = 7

    ##-------------------------------------------------------------
    d1 = digit1 + 1
    d2 = digit2 + 1

    # """Obtaining the data"""
    print("Getting data...")
    mnist = loadmat('datasets/MNISTmini.mat')
    x_train, y_train, x_test, y_test = extractMNISTmini(mnist, 'train_fea1', 'train_gnd1', 'test_fea1', 'test_gnd1')

    # """Obtaining subset of data for digits 4 and 7""" 
    digit1train, digit2train, digit1gnd, digit2gnd = divideDigitData(d1, d2, x_train, y_train)                         

    # """Creating training, validation, and test sets"""
    print("Creating train/val/test sets...")
    dataX, dataY = combineDigitData(digit1train, digit2train, digit1gnd, digit2gnd)
    x_train, x_valid, y_train, y_valid = train_test_split(dataX, dataY, train_size=0.5)
    
    x_train = DataLoader(torch.from_numpy(x_train))
    print(x_train)
    y_train = DataLoader(torch.from_numpy(y_train))
    x_valid = DataLoader(torch.from_numpy(x_valid))
    y_valid = DataLoader(torch.from_numpy(y_valid))

    print("Creating tests sets...")
    digit1test, digit2test, digit1gnd, digit2gnd = divideDigitData(d1, d2, x_test, y_test)                         
    dataXtest, dataYtest = combineDigitData(digit1test, digit2test, digit1gnd, digit2gnd)
    x_dummy, x_test, y_dummy, y_test = train_test_split(dataXtest, dataYtest, test_size=0.99)
    
    x_dummy = DataLoader(torch.from_numpy(x_dummy))
    y_dummy = DataLoader(torch.from_numpy(y_dummy))
    x_test = DataLoader(torch.from_numpy(x_test))
    y_test = DataLoader(torch.from_numpy(y_test))

    net = Net()
    if case == 0:
        if torch.cuda.is_available():
            net = net.cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)

        epochs = 5

        for e in range(epochs):
            train_loss = 0.0
            for data, labels in tqdm(x_train):
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

    if case == 1:
        print("Hello World")

if __name__== "__main__":
    main()