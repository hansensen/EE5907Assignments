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
import scipy.stats

#%%
# Load data from spamData.mat
xtrain, ytrain, xtest, ytest = du.loadData('spamData.mat')
# Log-transformation
xtrain = np.log(xtrain + 0.00001)
xtest = np.log(xtest + 0.00001)

#%%
    
def getGaussianNbClassifier(xtrain, ytrain):
    # Get ML Estimation of mu, variance and lambda
    mu, var = du.getMuVarMl(xtrain)
    lambdaMl = du.getLambdaML(xtrain)
    print('muMl: ', mu)
    print('varMl: ', var)
    print('lambdaMl: ', lambdaMl)

    # Get an array of unique classes, C
    classes = np.unique(ytrain)
    print('classes: ', classes)

    # Iterate by classes 0 and 1
    for i in range(len(classes)):
        # First term: logP(y = i | lambdaML)
        if classes[i]:
            logP_yTildeI = np.log(lambdaMl)
        else:
            logP_yTildeI = np.log(1 - lambdaMl)

        # Following terms: sum of logP(xTildej | xi <-c,j, yTilde = c)
        ytrain = ytrain.flatten()
        xClass = xtrain[ytrain == classes[i]]
        
    return 

print(getGaussianNbClassifier(xtrain, ytrain))

#%%
def getPredictions(classifier, xtest):
    return

#%%
def calcPosteriorProdictiveDist(gaussianNaiveBayesClassifier, xtest):
    return np.array()
    

#%%
def getErrorRate(predictedRes, yActual):
    predictedRes = predictedRes.astype(int)
    yActual = yActual.flatten().astype(int)
    return np.mean( predictedRes != yActual )

#%%
# Get classifier
gaussianNaiveBayesClassifier = getGaussianNbClassifier(xtrain, ytrain)

#%%
# Get error rate on training data
predictedTrainY = calcPosteriorProdictiveDist(gaussianNaiveBayesClassifier, xtest)
errRateTrain = getErrorRate(predictedTrainY, ytrain)
print('Error Rate on Training Data: ', errRateTrain)

# Get error rate on test data
predictedTestY = calcPosteriorProdictiveDist(gaussianNaiveBayesClassifier, xtest)
errRateTest = getErrorRate(predictedTestY, ytest)
print('Error Rate on Test Data: ', errRateTest)


