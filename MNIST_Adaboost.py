from scipy.io import loadmat
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# sklearn datasets are stored as numpy arrays

def main():

    print("Getting data...")
    feature = "Pixel"       # Pixel or LeNet5
    if feature == "Pixel":
        print(f'\t using Pixel features')
        MNIST = loadmat('datasets/MNIST.mat')
        # Separate the data, here they are as <class 'numpy.ndarray'>
        X_train = MNIST['train_fea']
        y_train = np.ravel(MNIST['train_gnd'])
        X_test = MNIST['test_fea']
        y_test = np.ravel(MNIST['test_gnd'])

        X_valid = X_train[50001:60000,]
        X_train = X_train[:50000,]
        y_valid = y_train[50001:60000]
        y_train = y_train[:50000]

    elif feature == "LeNet5":
        print(f'\t using LeNet5 features')
        LeNet5 = loadmat('datasets/MNIST-LeNet5.mat')
        # Separate the data, here they are as <class 'numpy.ndarray'>
        X_train = LeNet5['train_fea']
        y_train = np.ravel(LeNet5['train_gnd'])
        X_test = LeNet5['test_fea']
        y_test = np.ravel(LeNet5['test_gnd'])

        X_valid = X_train[50001:60000,]
        X_train = X_train[:50000,]
        y_valid = y_train[50001:60000]
        y_train = y_train[:50000]

    print("Setting up the model..")
    ada = AdaBoostClassifier(base_estimator=None,
                             n_estimators=50,
                             learning_rate=1.0,
                             algorithm='SAMME.R',
                             random_state=None)
    
    print("Training the model...")
    model = ada.fit(X_train, y_train)
    preds = ada.predict(X_test)
    avg_accuracy = ada.score(X_test, y_test)
    print(avg_accuracy)

    print("Plotting Average accuracy vs # of trees...")
    plt.plot(avg_accuracy, 50)
    #cv_params = {}


if __name__== "__main__":
    main()