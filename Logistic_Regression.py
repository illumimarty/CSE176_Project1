import pandas as pd
from sklearn.linear_model import LogisticRegression

train_fea1 = pd.read_csv('./datasets/train_fea1.csv')
train_gnd1 = pd.read_csv('./datasets/train_gnd1.csv')

# Train on digit 1 and 7 class(2 and 8)

# Setup y
y_digit1 = train_gnd1[train_gnd1['1'] == 2]
y_digit7 = train_gnd1[train_gnd1['1'] == 8]
train_y = pd.concat([y_digit1, y_digit7])

# Setup X
# Digit 1 row: 6741 - 12698
x_digit1 = train_fea1[6741:12699]
# Digit 7  row: 42276 - 48126
x_digit7 = train_fea1[42276:48127]
train_x = pd.concat([x_digit1, x_digit7])

# Since we are required to use L2 regularization, both version of the following model work 
model = LogisticRegression(solver='newton-cg')
#model = LogisticRegression(solver='sag', max_iter=600)
model.fit(train_x, train_y.values.ravel())