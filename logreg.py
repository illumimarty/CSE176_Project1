from os.path import dirname, join as pjoin
import pandas as pd
import scipy.io as sio
import os

# data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
# script_dir = os.path.dirname(__file__)
# rel_path = "MNISTmini.mat"
# mat_fname = os.path.join(script_dir, rel_path)

# mat_contents = sio.loadmat(mat_fname)
# traindata = mat_contents['train_fea1']

# for i in range(100):
#     columns
    # print('x' + str(i))



# print(traindata.tolist())


# f = open("test.txt", "a")
# f.write(str(traindata.tolist()))
# f.close()

# df = pd.read_csv('datasets/train_fea1.csv')
# print(df.to_string())

# f = open("text.txt", "r")
# print(f.read())

train_x = pd.read_csv('./datasets/train_fea1.csv')
train_y = pd.read_csv('./datasets/train_gnd1.csv')
print(train_x)