from re import I
from scipy.io import loadmat
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
# sklearn datasets are stored as numpy arrays

def main():

    print("Getting data...")
    feature = "LeNet5"       # Pixel or LeNet5
    if feature == "Pixel":
        print(f'\t using Pixel features')
        MNIST = loadmat('datasets/MNIST.mat')
        # Separate the data, here they are as <class 'numpy.ndarray'>
        X_train = MNIST['train_fea']
        y_train = np.ravel(MNIST['train_gnd'])
        X_test = MNIST['test_fea']
        y_test = np.ravel(MNIST['test_gnd'])

        X_train, X_valid = train_test_split(X_train, test_size = 0.16, train_size = 0.84, shuffle = True)
        y_train, y_valid = train_test_split(y_train, test_size = 0.16, train_size = 0.84, shuffle = True)

    elif feature == "LeNet5":
        print(f'\t using LeNet5 features')
        LeNet5 = loadmat('datasets/MNIST-LeNet5.mat')
        # Separate the data, here they are as <class 'numpy.ndarray'>
        X_train = LeNet5['train_fea']
        y_train = np.ravel(LeNet5['train_gnd'])
        X_test = LeNet5['test_fea']
        y_test = np.ravel(LeNet5['test_gnd'])
        
        X_train, X_valid = train_test_split(X_train, test_size = 0.16, train_size = 0.84, shuffle = True)
        y_train, y_valid = train_test_split(y_train, test_size = 0.16, train_size = 0.84, shuffle = True)
        print(X_train)

    # print("Setting up the model..")
    # ada = AdaBoostClassifier(base_estimator=None,
    #                          n_estimators=50,
    #                          learning_rate=1.0,
    #                          algorithm='SAMME.R',
    #                          random_state=None)
    
    # print("Training the model...")
    # model = ada.fit(X_train, y_train)
    # preds = ada.predict(X_test)     # These are the predicted labels of X_test
    # avg_accuracy = ada.score(X_test, y_test)

    # loss = 0
    # total = 0
    # for i in range(len(y_test)):
    #     total += 1
    #     if y_test[i] != preds[i]:
    #         loss += 1
    
    # MSE = loss / total

    print("Cross Validating...")
    NE_range = [25, 50, 75, 100, 125, 300, 500]
    LR_range = [0.0001, 0.001, 0.01, 0.1 , 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0]
    MD_range = [1, 2, 3, 4, 5, 6, 7, 8]
    for i in NE_range:
        ada = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 3),
                                 n_estimators=i,
                                 learning_rate=0.3,
                                 algorithm='SAMME.R',
                                 random_state=None)
        model = ada.fit(X_train, y_train)
        preds = ada.predict(X_valid)

        loss = 0
        total = 0
        for j in range(len(y_valid)):
            total += 1
            if y_valid[j] != preds[j]:
                loss += 1
        
        MSE = loss / total
        acc = ada.score(X_valid, y_valid)
        train_err = ada.score(X_train, y_train)
        print(f'n_estimators = {i} | Error: {MSE} | Accuracy: {acc} | Train Accuracy: {train_err}')

    # print("Plotting Average accuracy vs # of trees...")
    # plt.plot(NE_range, mean_err,
    #          linewidth = 2)
    # plt.title("Mean Accuracy based on Learning Rate")
    # plt.xlabel("Learning Rate")
    # plt.ylabel("Mean Accuracy")
    # plt.show()
    #plt.savefig(string[filename])



if __name__== "__main__":
    main()