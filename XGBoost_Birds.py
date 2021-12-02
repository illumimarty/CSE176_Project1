
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
import pandas as pd
import seaborn as sns
from clean import createImageDataset, getFeaturesAndLabels, getDigitGnd, getDigitRange
from sklearn.metrics import accuracy_score

def saveModel(model):
    i = 1
    model_dir = "./models/"
    json_dir = "./json/"
    prefix = "xgb-clf-"
    suffix = ".model"

    model_version = model_dir + prefix + str(i) + suffix
    json_version = json_dir + prefix + str(i)

    while os.path.exists(model_version) is True:
        i += 1
        model_version = prefix + str(i) + suffix

    model.save_model(model_version)
    model.dump_model(json_version, dump_format="json")

def main():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # removes a warning when using tensorflow

    # download dataset from Kaggle, unzip and place in /datasets folder with following name
    train_data_dir = "./datasets/100-bird-species/train"
    valid_data_dir = "./datasets/100-bird-species/valid"
    # test_data_dir = "./datasets/100-bird-species/test"

    """DATA EXTRACTION PARAMETERS"""
    train_batch_size = 1000    # 45980 images in training
    valid_batch_size = 35       # 1575 images in validation
    # test_batch_size = 1         # -- images in testing
    seed_val = 42
    isColored = False           # if False -> color_mode = "grayscale", meaning way less features
    isNormalized = True         # for pixel brightness: if False -> [0,255], elif -> [0,1]

    """MODEL PARAMETERS"""
    num_trees = 3
    param = {
        'objective': 'multi:softmax',
        # 'num_parallel_tree': 3,
        'subsample': 0.8,
        'num_class': 7,
        'max_depth': 3,
        'tree_method': 'hist',
        'random_state': 42,
        'eval_metric': 'mlogloss'
    }

    

    # ---------------------------------------------------------------------------
    """Data Extraction and Preprocessing"""
    train_set = createImageDataset(batchSize=train_batch_size, path=train_data_dir, color=isColored, seedVal=seed_val)
    valid_set = createImageDataset(batchSize=valid_batch_size, path=valid_data_dir, color=isColored, seedVal=seed_val)
    # test_set = createImageDataset(batchSize=test_batch_size, path=test_data_dir, color=isColored, seedVal=seed_val)

    train_features, train_labels = getFeaturesAndLabels(norm=isNormalized, batch=train_set)
    valid_features, valid_labels = getFeaturesAndLabels(norm=isNormalized, batch=valid_set)
    # test_features, test_labels = getFeaturesAndLabels(norm=isNormalized, batch=test_set)

    # DON'T USE YET, MIGHT BE USEFUL LATER
    # --------------------------------------------

    # classSix = getDigitRange(0, train_labels)
    # for i in classSix:
    #     print(i)
 

    # train_fea_df = pd.DataFrame(train_features, columns=col)
    # train_gnd_df = pd.DataFrame(train_labels, columns=['species'])
    # valid_fea_df = pd.DataFrame(valid_features, columns=col)
    # valid_gnd_df = pd.DataFrame(valid_labels)
    # print(train_fea_df.head())
    # print(train_fea_df.tail())
    # print(train_gnd_df.head())
    # print(train_gnd_df.tail())

    # sns.countplot(x='species', data=train_gnd_df)
    # plt.show()

    # --------------------------------------------

    print("Creating model...")
    dtrain = xgb.DMatrix(data=train_features, label=train_labels)

    print("Training model...")
    start = time.time()
    gbm = xgb.train(param, dtrain, num_trees)
    end = time.time()
    saveModel(gbm)
    print("Training complete. Elapsed time in seconds: " + str(end-start))

    print("Predicting...")
    dvalid_fea = xgb.DMatrix(valid_features)
    y_pred = gbm.predict(dvalid_fea)

    pred = [round(value) for value in y_pred]
    accuracy = accuracy_score(valid_labels, pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

if __name__== "__main__":
    main()
