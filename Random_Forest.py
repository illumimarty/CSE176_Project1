import pandas as pd
import numpy as np
import seaborn as sns
from clean import *
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, validation_curve
import matplotlib.pyplot as plt
from math import exp
import time

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
    print("Performing cross-validation...")
    depthList = np.linspace(5, 100, 10)
    # clist = np.linspace(1,-1, 20)
    # clist = np.logsp

    ## ADD PARAMETERS HERE
    model = RandomForestClassifier()
    start_time = time.time()
    train_score, test_score = validation_curve(model, x_train, y_train,
                                            param_name="max_depth",
                                            param_range=depthList,
                                            scoring="accuracy",
                                            cv=5)
    end_time = time.time()
    print("Cross-validation time: " + str(round((end_time-start_time), 2)) + "s")
    # Calculating mean and standard deviation of training score
    mean_train_score = np.mean(train_score, axis = 1)
    
    # Calculating mean and standard deviation of testing score
    mean_test_score = np.mean(test_score, axis = 1)
    
    # Plot mean accuracy scores for training and testing scores
    plt.semilogx(depthList, mean_train_score, label = "Training Score", color = 'b')
    plt.semilogx(depthList, mean_test_score, label = "Cross Validation Score", color = 'g')
    
    # Creating the plot
    plt.title("Validation Curve with Random Forests")
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    # plt.xlim(10e-10, 10e-4)
    # plt.ylim(0.0, 1.1)
    plt.legend(loc = 'best')
    plt.show()

    # """Creating, fitting, and making predictions with the model"""
    # ## comment out models to test out
    # print("Fitting model...")
    # model = RandomForestClassifier(max_depth=35, random_state=0)
    # model.fit(x_train, y_train)


    # print("Making predictions...")
    # y_pred = model.predict(x_test)

    # print("[Accuracy]",metrics.accuracy_score(y_test, y_pred))

    # """Showcasing accuracy via confusion matrix"""
    # cm = confusion_matrix(y_pred, y_test)

    # # print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # score = model.score(x_test, y_test)

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
    # ## End comment


    # """Visualize misclassified images"""
    # index = 0
    # misclassifiedIndexes = []
    # for label, predict in list(zip(y_pred, y_test)):
    #     index +=1
    #     if label != predict: 
    #         misclassifiedIndexes.append(index)

    # plt.figure(figsize=(20,4))
    # for plotIndex, badIndex in enumerate(misclassifiedIndexes[1:5]):
    #     plt.subplot(1, 5, plotIndex + 1)
    #     plt.imshow(np.reshape(x_test[badIndex], (10,10)), cmap=plt.cm.gray)
    #     # plt.title("Predicted: {}, Actual: {}".format(y_test[badIndex], x_test[badIndex]), fontsize = 15)
    # plt.show()





if __name__== "__main__":
    main()