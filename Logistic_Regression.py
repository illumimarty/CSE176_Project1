import pandas as pd
import numpy as np
import seaborn as sns
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

## creating functions to clean up code
# def getTrainData(fea, dng, digit):
#
def getDigitRange(gnd, digit):
    digitrange = np.where(np.logical_and(gnd > digit-1, gnd < digit+1))
    digitidx = digitrange[0]
    return digitidx

def main(): 
    """Obtaining the data"""
    print("Getting data...")
    mnist = loadmat('datasets/MNISTmini.mat')
    x_train = np.array(mnist['train_fea1'])
    y_train = np.array(mnist['train_gnd1'])
    x_test = np.array(mnist['test_fea1'])
    y_test = np.array(mnist['test_gnd1'])

    ## EDIT THESE 2 VARS TO CHANGE DIGIT CLASSES ##
    digit1 = 4
    digit2 = 9

    d1 = digit1 + 1
    d2 = digit2 + 1

    """Obtaining subset of data for digits 4 and 7"""
    digit1idx = getDigitRange(y_train, d1)                              # list of indexes
    digit1train = x_train[digit1idx]                                    # extract features based on indexes
    digit1gnd = y_train[digit1idx].flatten()                            # extract ground trurth based on indexes

    digit2idx = getDigitRange(y_train, d2)
    digit2train = x_train[digit2idx]
    digit2gnd = y_train[digit2idx].flatten()

    """Creating training set"""
    x_train = np.concatenate((digit1train, digit2train))
    y_train = np.concatenate([digit1gnd, digit2gnd])

    """Creating, fitting, and making predictions with the model"""
    ## comment out models to test out
    model = LogisticRegression(solver='liblinear', random_state=0, max_iter=1000)
    # model = LogisticRegression(solver='sag', max_iter=1000)

    print("Fitting model...")
    model.fit(x_train, y_train)

    ## Creating the test/validation sets
    digit1idx = getDigitRange(y_test, d1)                              
    digit1test = x_test[digit1idx]
    digit1gnd = y_test[digit1idx].flatten() 

    digit2idx = getDigitRange(y_test, d2)                                      
    digit2test = x_test[digit2idx]
    digit2gnd = y_test[digit2idx].flatten() 

    x_test = np.concatenate((digit1test, digit2test))
    y_test = np.concatenate((digit1gnd, digit2gnd))

    print("Making predictions...")
    y_pred = model.predict(x_test)

    """Showcasing accuracy via confusion matrix"""
    cm = confusion_matrix(y_pred, y_test)

    # print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    score = model.score(x_test, y_test)

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

    ## start comment
    """Visualize misclassified images"""
    index = 0
    misclassifiedIndexes = []

    # with open('y_pred.txt', 'w') as f:
    #     x = y_pred
    #     # print(x[0][1])
    #     for item in x:
    #         f.write(str(item) + "\n") 
    #     f.close()

    # with open('y_test.txt', 'w') as f:
    #     x = y_test
    #     # print(x[0][1])
    #     for item in x:
    #         f.write(str(item) + "\n") 
    #     f.close()
        
    # with open('misclf.txt', 'w') as f:
    #     i = 0
    #     x = list(zip(y_pred, y_test))
    #     # print(x[0][1])
    #     for predict, label in x:
    #         # f.write(str(item) + "\n")
    #         i+=1
    #         if predict != label:
    #             # f.write(str(predict) + "," + str(label)) 
    #             f.write(str(i) + "\n")
    #     f.close()    

    for label, predict in list(zip(y_pred, y_test)):
        index +=1
        if label != predict: 
            misclassifiedIndexes.append(index)

    # with open('test.txt', 'w') as f:
    #     x = misclassifiedIndexes
    #     # print(x[0][1])
    #     for item in x:
    #         f.write(str(item)) 
    #     f.close()

    plt.figure(figsize=(20,4))
    for plotIndex, badIndex in enumerate(misclassifiedIndexes[1:5]):
        plt.subplot(1, 5, plotIndex + 1)
        plt.imshow(np.reshape(x_test[badIndex], (10,10)), cmap=plt.cm.gray)
        # plt.title("Predicted: {}, Actual: {}".format(y_test[badIndex], x_test[badIndex]), fontsize = 15)
    plt.show()

    ## End comment

if __name__== "__main__":
  main()