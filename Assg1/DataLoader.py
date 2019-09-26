import scipy.io as scio
import numpy as np

def loadData(fileName):
    data = scio.loadmat(fileName)
    xtest = data['Xtest']
    xtrain = data['Xtrain']
    ytest = data['ytest']
    ytrain = data['ytrain']
    return xtrain, ytrain, xtest, ytest