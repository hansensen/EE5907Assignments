# %%
import scipy.io as sio
import numpy as np
import DataUtil as du
import math
import matplotlib.pyplot as plot
import scipy.stats as sst

np.seterr(divide='ignore')


# %%
# get an array of lambda
def getLambda():
    lambda1 = np.arange(1, 11, 1)
    lambda2 = np.arange(15, 105, 5)
    return np.concatenate((lambda1, lambda2), axis=0)


# %%
# Load data from spamData.mat
xtrain, ytrain, xtest, ytest = du.loadData('spamData.mat')
# Log-transformation
xtrain = np.log(xtrain + 0.1)
xtest = np.log(xtest + 0.1)
# Add bias term 1 to the start of x
numSample = xtrain.shape[0]
xtrainBias = np.concatenate((np.ones((numSample, 1)), xtrain), axis=1)
numSample = xtest.shape[0]
xtestBias = np.concatenate((np.ones((numSample, 1)), xtest), axis=1)

lambdaArr = getLambda()

trainErr = np.zeros(len(lambdaArr))
testErr = np.zeros(len(lambdaArr))


for i in range(len(lambdaArr)):
    lam = lambdaArr[i]


# %%
