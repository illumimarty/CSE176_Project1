import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

def plotCVCurves(params, train, test, model, hyperparam):
    # Plot mean accuracy scores for training and testing scores
    plt.semilogx(params, train, label = "Training Score", color = 'b')
    plt.semilogx(params, test, label = "Cross Validation Score", color = 'g')

    # Creating the plot
    plt.title("Validation Curve with " + model)
    plt.xlabel(hyperparam)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc = 'best')
    plt.show()    

def confustionMatrix(yPred, yTrain, digit1, digit2):
    """Showcasing accuracy via confusion matrix"""
    cm = confusion_matrix(yPred, yTrain)

    score = metrics.accuracy_score(yTrain, yPred)
    score = round(score*100, 4)

    labels = [digit1, digit2]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    # create heatmap
    sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    all_sample_title = 'Accuracy Score: {0}%'.format(score)
    plt.title(all_sample_title, size=15)
    plt.show()
