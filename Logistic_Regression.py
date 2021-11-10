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

## EDIT THESE 2 VARS TO CHANGE DIGIT CLASSES ##
digit1 = 4
digit2 = 9

d1 = digit1 + 1
d2 = digit2 + 1

l1 = d1-1
l2 = d2-1
r1 = d1+1
r2 = d2+1


"""Obtaining subset of data for digits 4 and 7"""
digit4range = np.where(np.logical_and(y_train > l1, y_train < r1))    # find indexes for specified digit
digit4idx = digit4range[0]                                          # list of indexes
digit4train = x_train[digit4idx]                                    # extract features based on indexes
digit4gnd = y_train[digit4idx].flatten()                            # extract ground trurth based on indexes

digit7range = np.where(np.logical_and(y_train > l2, y_train < r2))
digit7idx = digit7range[0]
digit7train = x_train[digit7idx]
digit7gnd = y_train[digit7idx].flatten()
# print(type(digit7gnd))

"""Creating training set"""
x_train = np.concatenate((digit4train, digit7train))
y_train = np.concatenate([digit4gnd, digit7gnd])

"""Creating, fitting, and making predictions with the model"""
## comment out models to test out
# model = LogisticRegression(solver='liblinear', random_state=0, max_iter=1000)
model = LogisticRegression(solver='sag', max_iter=1000)

print("Fitting model...")
model.fit(x_train, y_train)

## Creating the test/validation sets
digit4range = np.where(np.logical_and(y_test > l1, y_test < r1))
digit4idx = digit4range[0]
digit4test = x_test[digit4idx]
digit4gnd = y_test[digit4idx].flatten() 

digit7range = np.where(np.logical_and(y_test > l2, y_test < r2))
digit7idx = digit7range[0]
digit7test = x_test[digit7idx]
digit7gnd = y_test[digit7idx].flatten() 

x_test = np.concatenate((digit4test, digit7test))

y_test = np.concatenate((digit4gnd, digit7gnd))
print("Making predictions...")
y_pred = model.predict(x_test)

"""Showcasing accuracy via confusion matrix"""
# needs some tinkering
cm = confusion_matrix(y_pred, y_test)
# print(cm)

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