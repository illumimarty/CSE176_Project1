from numpy.lib.npyio import load
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

mnist = loadmat('MNISTmini.mat')

pd.options.display.max_columns = None
pd.options.display.max_rows = None

Xtrain = 'train_fea1'
# p = mnist[Xtrain][0].flat[38]
# p = [item.flat[0] for item in mnist[Xtrain][0]]
x_train = [[row.flat[0] for row in line] for line in mnist[Xtrain]]




# print(p)
# x_train = [[element for element in upperElement] for upperElement in mnist['train_fea1']]
numFea = len(x_train[0]) # of features = 100
x_col = []

for i in range(numFea):
    pixel = 'p' + str(i)
    x_col.append(pixel)
    # print(pixel)
x_digits4 = pd.DataFrame(x_train[24674:30094], columns = x_col)
x_digits7 = pd.DataFrame(x_train[42278:48128], columns = x_col)
# x_train = x_digits4.append(x_digits7)

with open('test.txt', 'w') as f:
    x = x_train
    f.write(str(x))
    f.close()
# df_train = pd.DataFrame(x_train, columns = x_col)
# print(df_train.head())












# print(df_train.loc[21])

# print(df_train)

# y_train = [[element for element in upperElement] for upperElement in mnist['train_gnd1']]

# print(mnist.keys())
# print(mnist['train_fea1'][1])
# print(str(item.flat[0] for item in mnist['train_fea1'][0])

# newData = list(zip(con))
# with open('test.txt', 'w') as f:
#     x = df_train
#     f.write(str(x))
#     f.close()