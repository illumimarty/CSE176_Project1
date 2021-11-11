import pandas as pd
import numpy as np
import seaborn as sns
from clean import *
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt

def main(): 

    ## EDIT THESE 2 VARS TO CHANGE DIGIT CLASSES ##
    digit1 = 4
    digit2 = 7

    d1 = digit1 + 1
    d2 = digit2 + 1

    """Obtaining the data"""
    print("Getting data...")
    mnist = loadmat('datasets/MNISTmini.mat')
    x_train = np.array(mnist['train_fea1'])
    y_train = np.array(mnist['train_gnd1'])
    x_test = np.array(mnist['test_fea1'])
    y_test = np.array(mnist['test_gnd1'])

    """Obtaining subset of data for digits 4 and 7"""                          
    digit1train = getDigitFea(d1, x_train, y_train)
    digit2train = getDigitFea(d2, x_train, y_train)
    digit1gnd = getDigitGnd(d1, y_train)
    digit2gnd = getDigitGnd(d2, y_train)

    """Creating training set"""
    x_train = np.concatenate((digit1train, digit2train))
    y_train = np.concatenate([digit1gnd, digit2gnd])

    """Creating, fitting, and making predictions with the model"""
    ## comment out models to test out
    # model = LogisticRegression(solver='liblinear', random_state=0, max_iter=1000)
    # model = LogisticRegression(solver='sag', max_iter=1000)

    ## Creating the test/validation sets
    digit1test = getDigitFea(d1, x_test, y_test)
    digit2test = getDigitFea(d2, x_test, y_test)
    digit1gnd = getDigitGnd(d1, y_test)
    digit2gnd = getDigitGnd(d2, y_test)

    x_test = np.concatenate((digit1test, digit2test))
    y_test = np.concatenate((digit1gnd, digit2gnd))

    """Perform cross-validation on the hyperparameter C"""
    clist = [round(i,2) for i in np.linspace(1,25,50)]
    train_acc = []
    valid_acc = []
    # print(clist)
    for i in range(50):
        print("Model ", i)
        # comment out models to test out
        # model = LogisticRegression(solver='liblinear', random_state=0, max_iter=1000, C = clist[i]).fit(x_train[:7000,:], y_train[:7000])
        model = LogisticRegression(solver='sag', max_iter=1000, C = clist[i]).fit(x_train[:7000,:], y_train[:7000])
        
        preds_train = model.predict(x_train[:7000,:])
        train_acc.append(accuracy_score(y_train[:7000], preds_train))

        preds_valid = model.predict(x_train[7001:,:])
        valid_acc.append(accuracy_score(y_train[7001:], preds_valid))

    plt.plot(clist, train_acc, label="Train")
    plt.plot(clist, valid_acc, label="Validation")
    plt.legend()
    plt.show()

if __name__== "__main__":
  main()