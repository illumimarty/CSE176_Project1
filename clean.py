import numpy as np

# extracts training and ground truth sets into
def extractMNISTmini(data, set1, set2, set3, set4): 
    xTrain = np.array(data[set1])
    yTrain = np.array(data[set2])
    xTest = np.array(data[set3])
    yTest = np.array(data[set4])
    return xTrain, yTrain, xTest, yTest

# concats the digit subsets into a single X input and y vector
def combineDigitData(d1fea, d2fea, d1gnd, d2gnd):
    X = np.concatenate((d1fea, d2fea))
    y = np.concatenate([d1gnd, d2gnd])
    return X, y

# for train/test sets, extract only the two specified digits
def divideDigitData(d1, d2, x, y):  
    d1fea = getDigitFea(d1, x, y)
    d2fea = getDigitFea(d2, x, y)
    d1gnd = getDigitGnd(d1, y)
    d2gnd = getDigitGnd(d2, y)
    return d1fea, d2fea, d1gnd, d2gnd

# get indexes of specified digit in MNISTmini
def getDigitRange(gnd, digit):      
    digit_range = np.where(np.logical_and(gnd > digit-1, gnd < digit+1))
    digit_idx = digit_range[0]
    return digit_idx

# extract features based on indexes
def getDigitFea(digit, fea, gnd):   
    digit_range = getDigitRange(gnd, digit)
    return fea[digit_range]

# extract ground truth based on indexes
def getDigitGnd(digit, gnd):        
    digit_range = getDigitRange(gnd, digit)
    return gnd[digit_range].flatten()