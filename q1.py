import scipy.io as sio
import numpy as np
from sklearn import preprocessing


def loadData(fileName):
	mat = sio.loadmat(fileName)
	xtest = mat['Xtest']
	xtrain = mat['Xtrain']
	ytest = mat['ytest']
	ytrain = mat['ytrain']
	return xtrain, ytrain, xtest, ytest

# Binarization preprocessing
def binarization(a):
	binarizer = preprocessing.Binarizer().fit(a)
	return binarizer.transform(a)

def getLambdaML(data):
	a = np.array(data)
	unique, counts = np.unique(a, return_counts=True)
	labels = dict(zip(unique, counts))
	# ML Estimate of lambda = N1/N
	return labels[1]/(labels[0]+labels[1])

def getNbClassifier(xtrain, ytrain):
	# lambdaMl = getLambdaML(ytrain)
	# Separate xtrain by class/label
	spam = []
	nonspam = []
	for i in range(len(xtrain)):
		vector = xtrain[i]
		if ytrain[i] == 0:
			nonspam.append(vector)
		else:
			spam.append(vector)

def getPredictions(classifier, xtest):


def main():
	# Load data from spamData.mat
	xtrain, ytrain, xtest, ytest = loadData('./spamData.mat')

	# Get classifier
	naiveBayesClassifier = getNbClassifier(xtrain, ytrain)

	# test model
	predictions = getPredictions(summaries, xtest)
	accuracy = getAccuracy(ytest, predictions)
	print('Accuracy: {0}%').format(accuracy)