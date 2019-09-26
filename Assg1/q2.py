# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'EE5907Assignments/Assg1'))
    print(os.getcwd())
except:
    pass

#%%
import scipy.io as sio
import numpy as np
import DataUtil as du
import math
import matplotlib.pyplot as plot
import scipy.stats as sst

#%%
# Load data from spamData.mat
xtrain, ytrain, xtest, ytest = du.loadData('spamData.mat')
# Log-transformation
xtrain = np.log(xtrain + 0.00001)
xtest = np.log(xtest + 0.00001)

#%%
    
def calcPosteriorProdictiveDist(xtrain, ytrain, xtest):
    # Get ML Estimation of mu, variance and lambda
    lambdaMl, N1, N = du.getLambdaML(ytrain)
    print('lambdaMl: ', lambdaMl)

    # Get an array of unique classes, C
    classes = [0, 1]
    print('classes: ', classes)

    # Init logP(y = c | x, D) array, index being c
    logP = []
    # Iterate by classes 0 and 1
    for i in range(len(classes)):
        # First term: logP(y = i | lambdaML)
        if classes[i]:
            logPyTildeI = np.log(lambdaMl)
        else:
            logPyTildeI = np.log(1 - lambdaMl)

        # Following terms: sum of logP(xTildej | xi <-c,j, yTilde = c)
        ytrain = ytrain.flatten()
        # Find all x samples with y being labeled as class i
        xtrainClassI = xtrain[ytrain == classes[i]]
        # From xtrainClassI data, get their mu and variance respectively
        # mu and var are 1D arrays with size same as no. of features
        mu, var = du.getMuVarMl(xtrainClassI)

        # Get log of P(xTildej | xi <-c,j, yTilde = c)
        logPxTilde = np.log(sst.norm(mu, np.sqrt(var)).pdf(xtest) + 0.000000001)

        # print('logP: ', logPyTildeI + np.sum(logPxTilde, axis = 1))
        # print(logP)

        # Sum all the terms together
        logP.append(logPyTildeI + np.sum(logPxTilde, axis = 1))

    return  logP

#%%
def getPredictions(classifier, xtest):
    return

#%%
def getErrorRate(predictedRes, yActual):
    predictedRes = predictedRes.astype(int)
    yActual = yActual.flatten().astype(int)
    return np.mean( predictedRes != yActual )

#%%
# Get error rate on training data
predictedTrainY = calcPosteriorProdictiveDist(xtrain, ytrain, xtest)
errRateTrain = getErrorRate(predictedTrainY, ytrain)
print('Error Rate on Training Data: ', errRateTrain)

# Get error rate on test data
predictedTestY = calcPosteriorProdictiveDist(xtrain, ytrain, xtest)
errRateTest = getErrorRate(predictedTestY, ytest)
print('Error Rate on Test Data: ', errRateTest)


