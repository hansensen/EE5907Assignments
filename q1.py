import scipy.io as sio
import numpy as np
from sklearn import preprocessing
import DataLoader as dl

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
	spam = np.array(spam).astype(int)
	nonspam = np.array(nonspam).astype(int)
	# calculate Xcj matrix
	numFeatures = spam.shape[1]
	print('Feauture Number: ', numFeatures)
	x = np.zeros((2, numFeatures)).astype(int)
	x[0] = np.sum(spam, axis=0)
	x[1] = np.sum(nonspam, axis=0)
	print(x[0])
	print(x[1])
	return x

def getPredictions(classifier, xtest):
	return

def main():
	# Load data from spamData.mat
	xtrain, ytrain, xtest, ytest = dl.loadData()
	xtrain = binarization(xtrain)
	xtest = binarization(xtest)

	print(xtrain)
	# Get classifier
	naiveBayesClassifier = getNbClassifier(xtrain, ytrain)

	# test model
	# predictions = getPredictions(summaries, xtest)
	# accuracy = getAccuracy(ytest, predictions)
	# print('Accuracy: {0}%').format(accuracy)

main()