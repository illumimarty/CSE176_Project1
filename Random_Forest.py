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

### CSE 176-01 - Fall 2021 Project1 - Random Forest Binary Classification on MNISTmini digits 4 and 7
### Andy Alvarenga, Moises Limon, Marthen Nodado, Ivan Palacios

# random.seed(68)

def main(): 
    """USER INPUT"""
    ## 0 - 5-fold cross-validation and adjusting hyperparameters
    ## 1 - Determining avg testing accuracy
    case = 0
    cm = True # if true and case=1, see confusion matrix

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
    x_train, x_valid, y_train, y_valid = train_test_split(dataX, dataY, train_size=0.99)

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


def crossValidation(xTrain, yTrain):
    print("Performing cross-validation...")
    depthList = np.linspace(100, 1500, 20, dtype=int)
    model = RandomForestClassifier()
    start_time = time.time()
    now = datetime.now()
    param = "n_estimators" # CHANGE MODEL PARAMATER HERE

    print("Cross-validation START:", now.strftime("%H:%M"))
    train_score, test_score = validation_curve(model, xTrain, yTrain,
                                            param_name=param, # DONT CHANGE IT HERE
                                            param_range=depthList,
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
    best_max_depth = depthList[best_idx]
    print("Best Accuracy:", mean_test_score[best_idx])
    print("Optimal max-depth hyperparameter:", best_max_depth)

    plotCVCurves(depthList, mean_train_score, mean_test_score, 'Random Forests', param)


def testAccuracy(xTrain, yTrain, xTest, yTest, cm):
    test_acc_list = []
    result_pred = []
    result_yTest = []

    for i in range(20):
        # print(i)
        model = RandomForestClassifier(max_depth=10, n_estimators=850)
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