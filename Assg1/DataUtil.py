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