import scipy.io as scio
import numpy as np
from sklearn import preprocessing


def loadData(fileName):
    data = scio.loadmat(fileName)
    xtest = data['Xtest']
    xtrain = data['Xtrain']
    ytest = data['ytest']
    ytrain = data['ytrain']
    return xtrain, ytrain, xtest, ytest

# Binarization preprocessing


def binarization(a):
    binarizer = preprocessing.Binarizer().fit(a)
    return binarizer.transform(a)

# Compute Maximum Likelihood Estimation of lambda


def getLambdaML(data):
    N1 = np.sum(data, axis=0)
    N = len(data)
    # ML Estimate of lambda = N1/N
    lambdaMl = N1 / N
    return lambdaMl

# Compute Maximum Likelihood Estimation of lambda


def getMuVarMl(data):
    mu = np.sum(data, axis=0) / len(data)
    variance = np.sum((data - mu) ** 2, axis=0) / len(data)
    return mu, variance
