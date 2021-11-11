import pandas as pd
import numpy as np
import seaborn as sns
from clean import *
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
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
    x_dummy, x_test, y_dummy, y_test = train_test_split(dataXtest, dataYtest, test_size=0.99)


    """Creating, fitting, and making predictions with the model"""
    ## comment out models to test out
    print("Initializing logreg model...")
    model = LogisticRegression(solver='liblinear', random_state=0, max_iter=1000)
    # model = LogisticRegression(solver='sag', max_iter=1000)

    print("Fitting model to training set...")
    model.fit(x_train, y_train)

    print("Making predictions...")
    y_pred = model.predict(x_train)

    """Showcasing accuracy via confusion matrix"""
    cm = confusion_matrix(y_pred, y_train)

    # # print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    score = metrics.accuracy_score(y_train, y_pred)
    score = round(score*100, 4)
    print("Accuracy:", score, "%")
    # score = model.score(y_train, y_pred)

    ## Start comment
    labels = [digit1, digit2]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    # create heatmap
    sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    # plt.title('Confusion matrix', y=1.1)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size=15)
    plt.show()
    ## End comment

    # ## start comment
    # """Visualize misclassified images"""
    # index = 0
    # misclassifiedIndexes = []

    # # with open('y_pred.txt', 'w') as f:
    # #     x = y_pred
    # #     # print(x[0][1])
    # #     for item in x:
    # #         f.write(str(item) + "\n") 
    # #     f.close()

    # # with open('y_test.txt', 'w') as f:
    # #     x = y_test
    # #     # print(x[0][1])
    # #     for item in x:
    # #         f.write(str(item) + "\n") 
    # #     f.close()
        
        preds_train = model.predict(x_train[:7000,:])
        train_acc.append(accuracy_score(y_train[:7000], preds_train))

        preds_valid = model.predict(x_train[7001:,:])
        valid_acc.append(accuracy_score(y_train[7001:], preds_valid))

    plt.plot(clist, train_acc, label="Train")
    plt.plot(clist, valid_acc, label="Validation")
    plt.legend()
    plt.show()

    ## End comment

if __name__== "__main__":
  main()