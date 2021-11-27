
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from tensorflow_datasets.core.dataset_utils import as_numpy
from xgboost import XGBClassifier
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import accuracy_score

def main():

    # download dataset from Kaggle, unzip and place in /datasets folder
    train_data_dir = "./datasets/100-bird-species/train"
    valid_data_dir = "./datasets/100-bird-species/valid"
    test_data_dir = "./datasets/100-bird-species/valid"

    # 45980 images in training
    # 1575 images in validation
    # -- images in testing

    train_batch_size = 32
    valid_batch_size = 32

    train_generator = image_dataset_from_directory(
        train_data_dir,
        labels="inferred",
        # color_mode="rgb",
        color_mode="grayscale",
        image_size=(224,224),
        shuffle="True",
        seed=42,
        batch_size=train_batch_size
    )

    valid_generator = image_dataset_from_directory(
        valid_data_dir,
        labels="inferred",
        color_mode="rgb",
        image_size=(224,224),
        shuffle="True",
        seed=42,
        batch_size=valid_batch_size
    )
    # test_generator = image_dataset_from_directory(
    #     test_data_dir,
    #     labels="inferred",
    #     color_mode="rgb",
    #     image_size=(224,224),
    #     shuffle="True",
    #     seed=42    
    # )

    # class_names = train_generator.class_names

    normalization_layer = layers.Rescaling(1./255)
    normalized_train_gen = train_generator.map(lambda x, y: (normalization_layer(x), y))
    normalized_valid_gen = valid_generator.map(lambda x, y: (normalization_layer(x), y))

    image_batch, labels_batch = next(iter(normalized_train_gen))
    train_features = as_numpy(image_batch)
    train_labels = as_numpy(labels_batch)

    image_batch, labels_batch = next(iter(normalized_valid_gen))
    valid_features = as_numpy(image_batch)
    valid_labels = as_numpy(labels_batch)
    

    print("Creating model...")
    model = XGBClassifier(
        objective="multi:softmax",
        max_depth=5,
        booster="gbtree",
        subsample=0.8,
        colsample_bynode="0.8",
        num_parallel_tree=100,
        n_estimators=1     #   n_boost_rounts
    )

    # not working yet,  need to figure out how to further preprocess the data
    print("Training model...")
    model.fit(train_features, train_labels)

    print("Predicting...")
    y_pred = model.predict(valid_features)
    pred = [round(value) for value in y_pred]

    accuracy = accuracy_score(valid_labels, pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))


    # dtrain = xgb.DMatrix(train_features, label=train_labels)
    # dval = xgb.DMatrix(valid_features)

    # print(dtrain[0])

    # param = { # required for random forest training
    #     'booster': 'gbtree',            # default value
    #     'subsample': 0.8,               # value < 1
    #     'colsample_bynode': 0.8,        # value < 1
    #     'num_parallel_tree': 100,       # size of forest
    #     'num_boost_round': 1,           # to prevent from boosting multiple random forests, set to 1
    #     'objective': 'multi:softmax',   # obj function: multiclass classification using softmax
    #     'max_depth': 5
    # }

    # num_round = 10

    # model = xgb.train(param, dtrain, num_round) # evallist for cross validation, to be used later
    # model.save_model('test_xgb.model')
    # model.dump_model('dump_test.raw.txt', 'featmap.txt')

    # ypred = model.predict(dval)

    # pred = [round(value) for value in ypred]
    # accuracy = accuracy_score(valid_labels, pred)
    # print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # image_batch, labels_batch = next(iter(normalized_train_gen))
    # first_image = image_batch[0]
    # print(class_names[labels_batch[0]])

if __name__== "__main__":
    main()
