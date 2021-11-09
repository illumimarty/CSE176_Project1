import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


pd.options.display.max_columns = None
pd.options.display.max_rows = None

train_fea1 = pd.read_csv('./datasets/train_fea1.csv')
train_gnd1 = pd.read_csv('./datasets/train_gnd1.csv')
test_fea1 = pd.read_csv('./datasets/test_fea1.csv')
test_gnd1 = pd.read_csv('./datasets/test_gnd1.csv')

# Train on digit 4 and 7 class(5 and 8)

# Setup y
y_digit4 = pd.DataFrame(train_gnd1[train_gnd1['1'] == 5])
y_digit7 = pd.DataFrame(train_gnd1[train_gnd1['1'] == 8])

train_y = y_digit4.append(y_digit7)
train_y.drop(train_y.tail(2).index,inplace=True) # drop last n rows


# with open('test.txt', 'w') as f:
#     x = train_y
#     # for output in x:
#     #     f.write(str(output))
#     f.write(str(x))
#     f.close()



# Setup X
# Digit 1 row: 6741 - 12698
x_digit4 = pd.DataFrame(train_fea1[24672:30092])
# Digit 7  row: 42276 - 48126
x_digit7 = pd.DataFrame(train_fea1[42276:48126])
# train_x = pd.concat([x_digit4, x_digit7])
train_x = x_digit4.append(x_digit7)

## test output if X and y work
# with open('test.txt', 'w') as f:
#     x = train_x
#     y = train_y
#     # for output in x:
#     #     f.write(str(output))
#     # f.write(str(x.columns.tolist()))
#     # f.write(str(y))
#     f.close()

# X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.25, random_state=123)

# Since we are required to use L2 regularization, both version of the following model work 
model = LogisticRegression(solver='newton-cg')
# model = LogisticRegression(solver='sag', max_iter=600)
model.fit(train_x, train_y.values.ravel())

# fixed data, now gotta show results and accuracy

x_digit4 = pd.DataFrame(test_fea1[4160:5051])
x_digit7 = pd.DataFrame(test_fea1[7038:8011])
test_x = x_digit4.append(x_digit7)

y_digit4 = pd.DataFrame(test_gnd1[test_gnd1['1'] == 5])
y_digit7 = pd.DataFrame(test_gnd1[test_gnd1['1'] == 8])


y_pred = pd.Series(model.predict(test_x))
y_test = pd.concat([y_digit4, y_digit7])

z = pd.concat([y_test, y_pred], axis=1)
z.columns = ['Truth', 'Prediction']
# z.head()


# model.predict_proba(train_x)

# y_digit4 = test_gnd1[test_gnd1['1'] == 5]
# y_digit7 = test_gnd1[test_gnd1['1'] == 8]
# test_y = pd.concat([y_digit4, y_digit7])

# x_digit4 = test_fea1[4160:5051]
# x_digit7 = test_fea1[7038:8011]
# test_x = pd.concat([x_digit4, x_digit7])

# print(model.predict_proba(test_x))


# y_pred = pd.Series(model.predict(test_x))
# y_test = y_test.reset_index