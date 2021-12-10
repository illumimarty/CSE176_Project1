from scipy.io import loadmat
import numpy as np
import xgboost as xgb

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


    # for col_name, label in y_train.iteritems():
    #     if True:
    #         y_train = label - 1
    #         break

    # param_dist = {'n_estimators':10,
    #             'use_label_encoder':False}

    # xgboost = xgb.XGBRFClassifier(**param_dist)

    # xgboost.fit(X_train, y_train, 
    #             eval_set=[(X_train, y_train), (X_test, y_test)],
    #             eval_metric='logloss',
    #             verbose=True)

    # evals_result = xgboost.evals_result()

if __name__== "__main__":
    main()