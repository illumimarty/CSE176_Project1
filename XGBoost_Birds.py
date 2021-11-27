
import os
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
import skimage
import tensorflow as tf
from tensorflow_datasets.core.dataset_utils import as_numpy
from xgboost import XGBClassifier
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, Normalizer

from skimage.feature import hog
from skimage.io import imread
from skimage.transform import rescale
from sklearn.base import BaseEstimator, TransformerMixin


class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """
    Convert an array of RGB images to grayscale
    """
 
    def __init__(self):
        pass
 
    def fit(self, X, y=None):
        """returns itself"""
        return self
 
    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.array([skimage.color.rgb2gray(img) for img in X])
     
 
class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """
 
    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X, y=None):
 
        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)
 
        try: # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])

def main():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # removes a warning when using tensorflow

    # download dataset from Kaggle, unzip and place in /datasets folder
    train_data_dir = "./datasets/100-bird-species/train"
    valid_data_dir = "./datasets/100-bird-species/valid"
    test_data_dir = "./datasets/100-bird-species/valid"

    # 45980 images in training
    # 1575 images in validation
    # -- images in testing

    train_batch_size = 45980
    # valid_batch_size = 32

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

    # valid_generator = image_dataset_from_directory(
    #     valid_data_dir,
    #     labels="inferred",
    #     # color_mode="rgb",
    #     color_mode="grayscale",
    #     image_size=(224,224),
    #     shuffle="True",
    #     seed=42,
    #     batch_size=valid_batch_size
    # )
    # test_generator = image_dataset_from_directory(
    #     test_data_dir,
    #     labels="inferred",
    #     color_mode="rgb",
    #     image_size=(224,224),
    #     shuffle="True",
    #     seed=42    
    # )

    # class_names = train_generator.class_names

    # normalization_layer = layers.Rescaling(1./255)
    # normalized_train_gen = train_generator.map(lambda x, y: (normalization_layer(x), y))
    # normalized_valid_gen = valid_generator.map(lambda x, y: (normalization_layer(x), y))

    normalized_train_gen = train_generator
    # normalized_valid_gen = valid_generator


    image_batch, labels_batch = next(iter(normalized_train_gen))
    train_features = as_numpy(image_batch)
    train_labels = as_numpy(labels_batch)

    # image_batch, labels_batch = next(iter(normalized_valid_gen))
    # valid_features = as_numpy(image_batch)
    # valid_labels = as_numpy(labels_batch)

    train_features = train_features.reshape(train_features.shape[0], -1)
    # valid_features = valid_features.reshape(valid_features.shape[0], -1)


    # print(train_features.shape)
    
    # for i in range(0, 5):
    #     print(train_features[i])

    # grayify = RGB2GrayTransformer()
    # hogify = HogTransformer(
    #     pixels_per_cell=(14, 14), 
    #     cells_per_block=(2,2), 
    #     orientations=9, 
    #     block_norm='L2-Hys'
    # )
    # scalify = StandardScaler()

    # gray = grayify.fit_transform(train_features)
    # hog = hogify.fit_transform(gray)
    # x_train_prepared = scalify.fit_transform(hog)

    # print(x_train_prepared.shape)
    print("Creating model...")
    model = XGBClassifier(
        # use_label_encoder=False,
        objective="multi:softmax",
        max_depth=5,
        booster="gbtree",
        subsample=0.8,
        colsample_bynode="0.8",
        num_parallel_tree=100,
        n_estimators=1     #   n_boost_rounts
    )

    # finally got something to work, however the model is not predicting correctly
    # i'm thinking that the features (in pixels) are not matching up with the labels.
    # maybe I should organize the pixel values and pack them together in their own dictionary, so that it will match up with the right labels
    # try out print(train_features.shape) and print(train_labels.shape) to see what I mean
    
    print("Training model...")
    model.fit(train_features, train_labels)
    model.save_model('first_xgg.model')

    # print("Predicting...")
    # y_pred = model.predict(valid_features)
    # pred = [round(value) for value in y_pred]

    # accuracy = accuracy_score(valid_labels, pred)
    # print("Accuracy: %.2f%%" % (accuracy * 100.0))


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
