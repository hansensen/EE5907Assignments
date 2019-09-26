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
import DataLoader as dl
import math
import matplotlib.pyplot as plot


#%%
# Load data from spamData.mat
xtrain, ytrain, xtest, ytest = dl.loadData('spamData.mat')
# Log-transformation
xtrain = np.log(xtrain + 0.00001)
xtest = np.log(xtest + 0.00001)

#%%
# Compute Maximum Likelihood Estimation of lambda
def getMl(data):
    mu = np.sum(data, axis = 0) / len(data)
    variance = np.sum((data - mu) ** 2, axis = 0) / len(data)
    return mu, variance


#%%
    
def getGaussianNbClassifier(xtrain, ytrain):
    # Separate xtrain by class/label
    spam = []
    nonspam = []
    for i in range(len(xtrain)):
        vector = xtrain[i]
        if ytrain[i] == 0:
            nonspam.append(vector)
        else:
            spam.append(vector)
    muSpam = np.mean(spam, axis = 0)
    muNonspam = np.mean(nonspam, axis = 0)
    varSpam = np.var(spam, axis=0)
    varNonspam = np.var(nonspam, axis = 0)
    return muSpam, muNonspam, varSpam, varNonspam

print(getGaussianNbClassifier(xtrain, ytrain))

#%%
def getPredictions(classifier, xtest):
    return


#%%
# Compute Maximum Likelihood Estimation of mean and variance with given training data
mu, var = getMl(xtrain)
print('muMl: ', mu)
print('varMl: ', var)

#%%
# Get classifier
gaussianNaiveBayesClassifier = getGaussianNbClassifier(xtrain, ytrain)

#%%
def calcPosteriorProdictiveDist(alpha, naiveBayesClassifier, lambdaMl, featureVector):
    if logP_yTilde[0] > logP_yTilde[1]:
        return 0
    else:
        return 1
    

#%%

# Set hyperparameters
#alpha = alphaArr[j]
#print('alpha: ', alpha)
hyperParam = []
predictedRes = np.zeros(len(xtest))
for i in range(xtest.shape[0]):
    featureVector = xtest[i]
    predictedRes[i] = calcPosteriorProdictiveDist(hyperParam, gaussianNaiveBayesClassifier, mu, var, featureVector)
    #print('predicted: ', predictedRes[i])
predictedRes = predictedRes.astype(int)
ytest = ytest.flatten().astype(int)
errRate = np.mean( predictedRes != ytest )
print(errRate)


