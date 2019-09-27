# %%
import scipy.io as sio
import numpy as np
import DataUtil as du
import math
import matplotlib.pyplot as plot
import scipy.stats as sst
import math


np.seterr(divide='ignore')


# %%

# sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# get an array of lambda
def getLambda():
    lambda1 = np.arange(1, 11, 1)
    lambda2 = np.arange(15, 105, 5)
    return np.concatenate((lambda1, lambda2), axis=0)


# get gradient of NLL(wBold) without regularisation term
def getG(xtrainBias, wBold):
    mu = sigmoid(np.matmul(xtrainBias, wBold))
    G = np.matmul(xtrainBias.transpose(), (mu - ytrain))
    return G


# get Hession of NLL(wBold) without regularisation term
def getH(xtrainBias, wBold):
    # S = N x N diagonal matrix, where i-th diagonal is mu_i(1 - mu_i)
    mu = sigmoid(np.matmul(xtrainBias, wBold))
    S = np.diag(np.squeeze((mu * (1 - mu)), axis=1))
    # H = Xt x S x X
    H = np.matmul(xtrainBias.transpose(), S)
    H = np.matmul(H, xtrainBias)
    return H


# %%
# Newton's Method, get w to minimise NLL
def newtonsMethod(xtestBias, xtrainBias):
    # Get dimensions
    # N: # of samples
    N = len(xtrainBias)
    # D: # of features, xtrainBias has D + 1 columns due to bias term
    D = len(xtrainBias[0] - 1)
    # Initialisation, wBold is a (D+1) x 1 vector
    wBold = np.zeros((D + 1, 1))
    margin = 1000
    # Deploy Newton's method to obtain optimal w
    while (margin > 0.01):
        g = getG(xtrainBias, wBold)
        h = getH(xtrainBias, wBold)
    return w


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
    w = newtonsMethod(xtestBias, xtrainBias)


# %%
