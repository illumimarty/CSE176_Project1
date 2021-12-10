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
    feature = "Pixel"       # Pixel or LeNet5
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
    mean_err = []
    mean_acc =[]
    NE_range = [150, 250, 350]
    LR_range = np.arange(0.1, 1.1, 0.1)
    for i in LR_range:
        ada = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 5),
                                 n_estimators=50,
                                 learning_rate=i,
                                 algorithm='SAMME.R',
                                 random_state=None)
        model = ada.fit(X_train, y_train)
        cv_score = cross_val_score(model, X_valid, y_valid, n_jobs = 6)
        cv_err = 1 - cv_score
        mean_err.append(cv_err.mean())
        mean_acc.append(cv_score.mean())

        print(f'Learning Rate = {i} | Average Error: {cv_err.mean()} | Average Accuracy: {cv_score.mean()}')

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