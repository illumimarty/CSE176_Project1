import pandas as pd
import numpy as np
import seaborn as sns
from clean import *
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, validation_curve
import matplotlib.pyplot as plt
import random 

random.seed(68)

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
    dataX = np.concatenate((digit1train, digit2train))
    dataY = np.concatenate([digit1gnd, digit2gnd])

    x_train, x_valid, y_train, y_valid = train_test_split(dataX, dataY, train_size=0.5)

    """Prepping the Test set"""
    digit1test = getDigitFea(d1, x_test, y_test)
    digit2test = getDigitFea(d2, x_test, y_test)
    digit1gnd = getDigitGnd(d1, y_test)
    digit2gnd = getDigitGnd(d2, y_test)

    dataXtest = np.concatenate((digit1test, digit2test))
    dataYtest = np.concatenate((digit1gnd, digit2gnd))

    x_dummy, x_test, y_dummy, y_test = train_test_split(dataXtest, dataYtest, test_size=0.99)

    """Perform cross-validation on the hyperparameter tol"""
    print("Cross-validating...")
    tlist = np.logspace(-12, 4, 25)

    ## ADD PARAMETERS HERE
    # The optimal value for the tolerance is seen from 1e-6 to 1e-12
    model = LogisticRegression(solver="liblinear", max_iter=1000)
    train_score, test_score = validation_curve(model, x_train, y_train,
                                            param_name="tol",
                                            param_range=tlist,
                                            scoring="accuracy",
                                            cv=5)
    # Calculating mean of the training score
    mean_train_score = np.mean(train_score, axis = 1)
    
    # Calculating mean of the testing score
    mean_test_score = np.mean(test_score, axis = 1)

    # Find the value that gives the best accuracy
    best_idx = np.argmax(mean_test_score, axis=0)
    best_Tol = tlist[best_idx]
    print("Best Tol: ", best_Tol)
    print("Best Accuracy: ", mean_test_score[best_idx])

    # Plot mean accuracy scores for training and testing scores
    plt.semilogx(tlist, mean_train_score, label = "Training Score", color = 'b')
    plt.semilogx(tlist, mean_test_score, label = "Cross Validation Score", color = 'g')
    
    # Creating the plot
    plt.title("Validation Curve with Logistic Regression")
    plt.xlabel("Tol")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc = 'best')
    plt.show()

if __name__== "__main__":
  main()