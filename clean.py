import numpy as np

def getDigitRange(gnd, digit):      # get indexes of specified digit in MNISTmini
    digit_range = np.where(np.logical_and(gnd > digit-1, gnd < digit+1))
    digit_idx = digit_range[0]
    return digit_idx

def getDigitFea(digit, fea, gnd):   # extract features based on indexes
    digit_range = getDigitRange(gnd, digit)
    return fea[digit_range]

def getDigitGnd(digit, gnd):        # extract ground truth based on indexes
    digit_range = getDigitRange(gnd, digit)
    return gnd[digit_range].flatten()