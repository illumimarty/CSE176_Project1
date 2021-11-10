import pandas as pd
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

## creating functions to clean up code
# def getTrainData(fea, dng, digit):
#
# def getDigitRange(gnd, digit):
#     digitrange = np.where(np.logical_and(gnd > digit-1, gnd < digit+1))
#     digitidx = digitrange[0]
#     return digitidx

# def main(): 
"""Obtaining the data"""
mnist = loadmat('datasets/MNISTmini.mat')
x_train = np.array(mnist['train_fea1'])
y_train = np.array(mnist['train_gnd1'])
x_test = np.array(mnist['test_fea1'])
y_test = np.array(mnist['test_gnd1'])

"""Obtaining subset of data for digits 4 and 7"""
digit4range = np.where(np.logical_and(y_train > 4, y_train < 6))    # find indexes for specified digit
digit4idx = digit4range[0]                                          # list of indexes
digit4train = x_train[digit4idx]                                    # extract features based on indexes
digit4gnd = y_train[digit4idx].flatten()                            # extract ground trurth based on indexes

digit7range = np.where(np.logical_and(y_train > 7, y_train < 9))
digit7idx = digit7range[0]
digit7train = x_train[digit7idx]
digit7gnd = y_train[digit7idx].flatten()
# print(type(digit7gnd))

"""Creating training set"""
x_train = np.concatenate((digit4train, digit7train))
y_train = np.concatenate([digit4gnd, digit7gnd])

"""Creating, fitting, and making predictions with the model"""
## comment out models to test out
# model = LogisticRegression(solver='liblinear', random_state=0)
model = LogisticRegression(solver='sag', max_iter=600)

model.fit(x_train, y_train)

## Creating the test/validation sets
digit4range = np.where(np.logical_and(y_test > 4, y_test < 6))
digit4idx = digit4range[0]
digit4test = x_test[digit4idx]

digit7range = np.where(np.logical_and(y_test > 7, y_test < 9))
digit7idx = digit7range[0]
digit7test = x_test[digit7idx]

x_test = np.concatenate((digit4test, digit7test))

"""Showcasing accuracy via confusion matrix"""
# needs some tinkering
cm = confusion_matrix(y_train, model.predict(x_train))    
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(4, 7), ticklabels=('Predicted 4s', 'Predicted 7s'))
ax.yaxis.set(ticks=(4, 7), ticklabels=('Actual 4s', 'Actual 7s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()


## Misc. code:
# print(digit4)
# print(y_train[3])
# print(y_train.shape)
# print(model.classes_)

# with open('test.txt', 'w') as f:
#     x = classification_report
#     f.write(str(x))
#     f.close()

















# train_fea1 = pd.read_csv('./datasets/train_fea1.csv')
# train_gnd1 = pd.read_csv('./datasets/train_gnd1.csv')

# # Train on digit 1 and 7 class(2 and 8)

# # Setup y
# y_digit1 = train_gnd1[train_gnd1['1'] == 2]
# y_digit7 = train_gnd1[train_gnd1['1'] == 8]
# train_y = pd.concat([y_digit1, y_digit7])

# # Setup X
# # Digit 1 row: 6741 - 12698
# x_digit1 = train_fea1[6741:12699]
# # Digit 7  row: 42276 - 48126
# x_digit7 = train_fea1[42276:48127]
# train_x = pd.concat([x_digit1, x_digit7])

# # Since we are required to use L2 regularization, both version of the following model work 
# model = LogisticRegression(solver='newton-cg')
# #model = LogisticRegression(solver='sag', max_iter=600)
# model.fit(train_x, train_y.values.ravel())