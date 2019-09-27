# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

# %%
import scipy.io as sio
import numpy as np
from sklearn import preprocessing
import DataUtil as du
import math
import matplotlib.pyplot as plt

# %%


def calcPosteriorProdictiveDist(xtrain, ytrain, xtest, alpha):
    # Get ML Estimation of lambda
    lambdaMl = du.getLambdaML(ytrain)
    # print('lambda', lambdaMl)

    # Get an array of unique classes, C
    classes = [0, 1]

    # Init logP(y = c | x, D) array, index being c
    logP = []

    # Iterate by classes 0 and 1
    for i in range(len(classes)):
        # First term: logP(y = i | lambdaML)
        if classes[i]:
            logPyTildeI = np.log(lambdaMl)
        else:
            logPyTildeI = np.log(1 - lambdaMl)
        # print('classes[i]', classes[i])
        # Following terms: sum of logP(xTildej | xi <-c,j, yTilde = c)
        ytrain = ytrain.flatten()
        # Find all x samples with y being labeled as class i
        xtrainClassI = xtrain[ytrain == classes[i]]
        # print('x_label', xtrainClassI)
        # From xtrainClassI data, get their n1 and n respectively
        n1 = np.sum(xtrainClassI, axis=0)
        n = len(xtrainClassI)
        # print('n1', n1.shape)
        # print('n', n)
        # print('alpha', alpha)
        posterior = (n1 + alpha)/(n + alpha + alpha)
        # print('posterior', posterior)
        logPxTilde = np.log((xtest > 0) * posterior +
                            (xtest <= 0) * (1 - posterior))
        # print('logPxTilde', logPxTilde)

        # Sum all the terms together
        logP.append(logPyTildeI + np.sum(logPxTilde, axis=1))
    return np.array(logP).transpose()


def getErrorRate(logP, yActual):
    predictedRes = getPredictions(logP)
    yActual = yActual.flatten().astype(int)
    return np.mean(predictedRes != yActual)


def getPredictions(logP):
    predictedClass = np.argmax(logP, axis=1)
    return predictedClass


# %%

# Load data from spamData.mat
xtrain, ytrain, xtest, ytest = du.loadData('spamData.mat')
xtrain = du.binarization(xtrain)
xtest = du.binarization(xtest)

# Create an array of alpha values, from 0 to 100 with step size 0.5
alphaStart = 0
alphaEnd = 100
alphaStepSize = 0.5
alphaArr = np.arange(alphaStart, alphaEnd + alphaStepSize, alphaStepSize)

trainErr = np.zeros(len(alphaArr))
testErr = np.zeros(len(alphaArr))

# %%

for j in range(len(alphaArr)):
    alpha = alphaArr[j]
    logP = calcPosteriorProdictiveDist(xtrain, ytrain, xtrain, alpha)
    # print('logP', logP)
    err = getErrorRate(logP, ytrain)
    # print('trainErr', err)
    trainErr[j] = err

    logP = calcPosteriorProdictiveDist(xtrain, ytrain, xtest, alpha)
    err = getErrorRate(logP, ytest)
    # print('testErr', err)
    testErr[j] = err

# %%

# Plot graph: alpha vs error rate
plt.figure()
plt.plot(alphaArr, trainErr, 'green', label='train')
plt.plot(alphaArr, testErr, 'red', label='test')
plt.legend()
plt.title('Q1: Beta-binomial Naive Bayes')
plt.xlabel('alpha')
plt.ylabel('Error Rate')
plt.show()

# %%
# Traing and testing error rates for alpha = 1, 10, 100
for i in [1, 10, 100]:
    j = int(i/0.5)
    print('alpha =', int(alphaArr[j]))
    print('training error:', trainErr[j])
    print('testing error:', testErr[j])

# %%
