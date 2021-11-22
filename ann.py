import numpy as np
import time
import random
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import from_numpy,randn
from torch.utils.data import DataLoader, random_split
from clean import *
from plots import *
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, validation_curve
from datetime import datetime



### CSE 176-01 - Fall 2021 Project1 - Logistic Regression Binary Classification on MNISTmini digits 4 and 7
### Andy Alvarenga, Moises Limon, Marthen Nodado, Ivan Palacios

# random.seed(68)

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10*10, 50)
        self.fc2 = nn.Linear(50,2)
    def forward(self, x):
        x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

def main(): 
    """USER INPUT"""
    ## 0 - 5-fold cross-validation and adjusting hyperparameters
    ## 1 - Determining avg testing accuracy
    ## 2 - Run Neural Networks
    case = 2
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

    print("Creating tests sets...")
    digit1test, digit2test, digit1gnd, digit2gnd = divideDigitData(d1, d2, x_test, y_test)                         
    dataXtest, dataYtest = combineDigitData(digit1test, digit2test, digit1gnd, digit2gnd)

    x_dummy, x_test, y_dummy, y_test = train_test_split(dataXtest, dataYtest, test_size=0.99)


    if case == 0:
        """Perform cross-validation on the hyperparameter C"""
        crossValidation(x_train, y_train)

    if case == 1:
        avg, pred, gnd = testAccuracy(x_train, y_train, x_test, y_test, cm)
        print("Best accuracy for our model (on avg): " + str(round(avg*100, 4)) + "%")
        if cm is True:
          confustionMatrix(pred, gnd, digit1, digit2)

    if case == 2:
        # print("boom")
        # print("Creating neural network")
        model = Net()
        if torch.cuda.is_available():
            model = model.cuda()
        
        epochs = 5
        min_valid_loss = np.inf
        # criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(model.parameters(), lr = 0.01)
        x_train = from_numpy(x_train)
        # x_valid = from_numpy(x_valid)
        # trainloader = DataLoader(x_train, batch_size=5)
        # validloader = DataLoader(x_valid, batch_size=5)

        print("x_train size", str(x_train.shape))
        print("x_valid size", str(x_valid.shape))
        # print(model)
        # output = model()

        output = model(x_train)
        target = randn(10)
        target = target.view(1,-1)
        criterion = nn.MSELoss()

        loss = criterion(output, target)
        print(loss)


        # for e in range(epochs):
        #     train_loss = 0.0
        #     for data, labels in trainloader:
        #         # Transfer Data to GPU if available
        #         if torch.cuda.is_available():
        #             data, labels = data.cuda(), labels.cuda()
                
        #         # Clear the gradients
        #         optimizer.zero_grad()
        #         # Forward Pass
        #         target = model(data)
        #         # Find the Loss
        #         loss = criterion(target,labels)
        #         # Calculate gradients
        #         loss.backward()
        #         # Update Weights
        #         optimizer.step()
        #         # Calculate Loss
        #         train_loss += loss.item()
            
        #     valid_loss = 0.0
        #     model.eval()     # Optional when not using Model Specific layer
        #     for data, labels in validloader:
        #         # Transfer Data to GPU if available
        #         if torch.cuda.is_available():
        #             data, labels = data.cuda(), labels.cuda()
                
        #         # Forward Pass
        #         target = model(data)
        #         # Find the Loss
        #         loss = criterion(target,labels)
        #         # Calculate Loss
        #         valid_loss += loss.item()
        
        #     print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(trainloader)} \t\t Validation Loss: {valid_loss / len(validloader)}')
            
        #     if min_valid_loss > valid_loss:
        #         print(f'Validation Loss Decreased({min_valid_loss:.6f\
        #         }--->{valid_loss:.6f}) \t Saving The Model')
        #         min_valid_loss = valid_loss
                
        #         # Saving State Dict
        #         torch.save(model.state_dict(), 'saved_model.pth')        

def crossValidation(xTrain, yTrain):
    print("Performing cross-validation...")
    clist = np.logspace(-8.5,6,25)
    model = LogisticRegression(solver="liblinear", max_iter=1000, 
                                tol=1e-7, # 1e-7 for combined optimal
                                intercept_scaling=1)  # 2 for combined optimal
    start_time = time.time()
    now = datetime.now()
    param = "C" # CHANGE MODEL PARAMATER HERE

    print("Cross-validation START:", now.strftime("%H:%M"))
    train_score, test_score = validation_curve(model, xTrain, yTrain,
                                            param_name=param, # DONT CHANGE IT HERE
                                            param_range=clist,
                                            scoring="accuracy",
                                            cv=5)
    end_time = time.time()
    now = datetime.now()
    print("Cross-validation END:", now.strftime("%H:%M"))
    print("Elapsed Time: " + str(round((end_time-start_time), 2)) + "s")

    # Calculating mean training and testing scores
    mean_train_score = np.mean(train_score, axis = 1)    
    mean_test_score = np.mean(test_score, axis = 1)

    best_idx = np.argmax(mean_test_score, axis=0)
    best_C = clist[best_idx]
    print("Best Accuracy:", mean_test_score[best_idx])
    print("Optimal C value:", best_C)

    plotCVCurves(clist, mean_train_score, mean_test_score, "Logistic Regression", param)

    
def testAccuracy(xTrain, yTrain, xTest, yTest, cm):
    test_acc_list = []
    result_pred = []
    result_yTest = []
    for i in range(100):
        # print(i)
        model = LogisticRegression(solver="liblinear", max_iter=1000,
                                C=0.6,
                                tol=0.00000004,
                                intercept_scaling=1.75)
        model.fit(xTrain, yTrain)
        preds = model.predict(xTest)
        score = accuracy_score(yTest, preds)
        test_acc_list.append(score)

    if cm is True:
      result_pred = preds
      result_yTest = yTest

    return np.mean(test_acc_list, axis=0), result_pred, result_yTest

if __name__== "__main__":
  main()