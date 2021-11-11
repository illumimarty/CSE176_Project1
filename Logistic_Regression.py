import pandas as pd
import numpy as np
import seaborn as sns
from clean import *
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, validation_curve
import matplotlib.pyplot as plt
from math import exp

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

    """Perform cross-validation on the hyperparameter C"""
    clist = np.logspace(-6,-3.75,20)

    ## ADD PARAMETERS HERE
    model = LogisticRegression(solver="liblinear", max_iter=1000)
    train_score, test_score = validation_curve(model, x_train, y_train,
                                            param_name="C",
                                            param_range=clist,
                                            scoring="accuracy",
                                            cv=5)
    # Calculating mean and standard deviation of training score
    mean_train_score = np.mean(train_score, axis = 1)
    
    # Calculating mean and standard deviation of testing score
    mean_test_score = np.mean(test_score, axis = 1)
    
    # Plot mean accuracy scores for training and testing scores
    plt.plot(clist, mean_train_score, label = "Training Score", color = 'b')
    plt.plot(clist, mean_test_score, label = "Cross Validation Score", color = 'g')
    
    # Creating the plot
    plt.title("Validation Curve with Logistic Regression")
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc = 'best')
    plt.show()


    # """Creating, fitting, and making predictions with the model"""
    # ## comment out models to test out
    # print("Initializing logreg model...")
    # model = LogisticRegression(solver='liblinear', random_state=0, max_iter=1000)
    # # model = LogisticRegression(solver='sag', max_iter=1000)

    # print("Fitting model to training set...")
    # model.fit(x_train, y_train)

    # print("Making predictions...")
    # y_pred = model.predict(x_train)

    # """Showcasing accuracy via confusion matrix"""
    # cm = confusion_matrix(y_pred, y_train)

    # # # print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # score = metrics.accuracy_score(y_train, y_pred)
    # score = round(score*100, 4)
    # print("Accuracy:", score, "%")
    # # score = model.score(y_train, y_pred)

    # ## Start comment
    # labels = [digit1, digit2]
    # fig, ax = plt.subplots()
    # tick_marks = np.arange(len(labels))
    # plt.xticks(tick_marks, labels)
    # plt.yticks(tick_marks, labels)
    # # create heatmap
    # sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu", fmt='g')
    # ax.xaxis.set_label_position("top")
    # # plt.title('Confusion matrix', y=1.1)
    # plt.ylabel('True')
    # plt.xlabel('Predicted')
    # all_sample_title = 'Accuracy Score: {0}'.format(score)
    # plt.title(all_sample_title, size=15)
    # plt.show()
    ## End comment 
if __name__== "__main__":
  main()