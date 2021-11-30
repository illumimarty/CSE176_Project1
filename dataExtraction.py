import pandas as pd
import numpy as np
import sklearn.model_selection as sklm
import xgboost as xgb


# Regression is being done on this
# Value we are predicting is NAD which is the last col of values
# Only loading 10k of data. REMOVE nrows TO USE ALL 33K

print('Loading data')
data = pd.read_csv('datasets/twitterData.txt', sep = ',' , index_col = False, engine = 'python', nrows = 10000)

# Titles of the col so we know what is what
# Built in time periods that is why there are numbers

data.columns = ['NCD_0', 'NCD_1', 'NCD_2', 'NCD_3', 'NCD_4', 'NCD_5', 'NCD_6', 
                'AI_0', 'AI_1', 'AI_2', 'AI_3', 'AI_4', 'AI_5', 'AI_6',
                'AS(NA)_0', 'AS(NA)_1', 'AS(NA)_2', 'AS(NA)_3', 'AS(NA)_4' , 'AS(NA)_5', 'AS(NA)_6',
                'BL_0', 'BL_1', 'BL_2', 'BL_3', 'BL_4', 'BL_5', 'BL_6',
                'NAC_0', 'NAC_1', 'NAC_2', 'NAC_3', 'NAC_4', 'NAC_5', 'NAC_6',
                'AS(NAC)_0', 'AS(NAC)_1', 'AS(NAC)_2', 'AS(NAC)_3', 'AS(NAC)_4', 'AS(NAC)_5', 'AS(NAC)_6',
                'CS_0', 'CS_1', 'CS_2', 'CS_3', 'CS_4', 'CS_5', 'CS_6', 
                'AT_0', 'AT_1', 'AT_2', 'AT_3', 'AT_4', 'AT_5', 'AT_6',
                'NA_0', 'NA_1', 'NA_2', 'NA_3', 'NA_4', 'NA_5', 'NA_6',
                'ADL_0', 'ADL_1', 'ADL_2', 'ADL_3', 'ADL_4', 'ADL_5', 'ADL_6',
                'NAD_0', 'NAD_1', 'NAD_2', 'NAD_3', 'NAD_4', 'NAD_5', 'NAD_6',
                'NAD']

#Seperation of xdata and ydata

xData = data.iloc[:, :77]
yData = data.iloc[:, 77:]

# Create the splits


print('Creating training/validation/test sets')
x_train, x_val, y_train, y_val = sklm.train_test_split(xData, yData, test_size = 0.33, random_state = 45, shuffle=True)

print('Datasets created')

# Here we are going to use XGBoost

# XGB data structure taking in the pandas df using the training data
dtrain = xgb.DMatrix(data = x_train, label = y_train)

# Since this is a regression problem we are using XGBRegressor 

xgReg = xgb.XGBRegressor(objective = )


