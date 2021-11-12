import numpy as np
import time
import random
import datetime
import matplotlib.pyplot as plt
from clean import *
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, validation_curve

### CSE 176-01 - Fall 2021 Project1 - Random Forest Binary Classification on MNISTmini digits 4 and 7
### Andy Alvarenga, Moises Limon, Marthen Nodado, Ivan Palacios

# random.seed(68)

def main(): 
    """USER INPUT"""
    ## 0 - 5-fold cross-validation and adjusting hyperparameters
    ## 1 - Determining avg testing accuracy
    case = 1

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

    """Creating training, validation, and test sets"""
    ## Prepping data for training and validation sets
    print("Creating train/val sets...")
    dataX = np.concatenate((digit1train, digit2train))
    dataY = np.concatenate([digit1gnd, digit2gnd])

    x_train, x_valid, y_train, y_valid = train_test_split(dataX, dataY, train_size=0.5)

    ## Prepping data for test set
    print("Creating tests sets...")
    digit1test = getDigitFea(d1, x_test, y_test)
    digit2test = getDigitFea(d2, x_test, y_test)
    digit1gnd = getDigitGnd(d1, y_test)
    digit2gnd = getDigitGnd(d2, y_test)

    dataXtest = np.concatenate((digit1test, digit2test))
    dataYtest = np.concatenate([digit1gnd, digit2gnd])

    x_dummy, x_test, y_dummy, y_test = train_test_split(dataXtest, dataYtest, test_size=0.99)

    if case == 0:
        """Perform cross-validation on the hyperparameter C"""
        crossValidation(x_train, y_train)

    if case == 1:
        avg = testAccuracy(x_train, y_train, x_test, y_test)
        print("Best accuracy for our model (on avg): " + str(round(avg*100, 4)) + "%")


def plotCVCurves(params, train, test):
    # Plot mean accuracy scores for training and testing scores
    plt.semilogx(params, train, label = "Training Score", color = 'b')
    plt.semilogx(params, test, label = "Cross Validation Score", color = 'g')

    # Creating the plot
    plt.title("Validation Curve with Random Forests")
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc = 'best')
    plt.show()    


def crossValidation(xTrain, yTrain):
    print("Performing cross-validation...")
    depthList = np.linspace(5, 100, 10)
    model = RandomForestClassifier()
    start_time = time.time()
    now = datetime.now()

    print("Cross-validation START:", now.strftime("%H:%M"))
    train_score, test_score = validation_curve(model, xTrain, yTrain,
                                            param_name="max_depth",
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
    plotCVCurves(depthList, mean_train_score, mean_test_score)
    


def testAccuracy(xTrain, yTrain, xTest, yTest):
    test_acc_list = []
    for i in range(100):
        print(i)
        model = RandomForestClassifier(max_depth=10, n_estimators=850)
        model.fit(xTrain, yTrain)
        preds = model.predict(xTest)
        score = accuracy_score(yTest, preds)
        # print("The accuracy of our best model: ", score)
        test_acc_list.append(score)

    return np.mean(test_acc_list, axis=0)


if __name__== "__main__":
    main()