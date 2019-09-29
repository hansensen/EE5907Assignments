
# %%
import scipy.io as sio
import numpy as np
import DataUtil as du
import matplotlib.pyplot as plot
import scipy.stats as sst
import matplotlib.pyplot as plt
np.seterr(divide='ignore')

# get an array of lambda
def getK():
    lambda1 = np.arange(1, 11, 1)
    lambda2 = np.arange(15, 105, 5)
    return np.concatenate((lambda1, lambda2), axis=0)

def getDistance(xtrain, x):
    # dis is a NxM matrix
    # N denotes the # of x samples
    N  = x.shape[0]
    #print(x.shape)
    # M denotes the # of xtrain samples
    M = xtrain.shape[0]
    dist = np.zeros((N, M))

    # dist[i,j] is the distance between i-th sample in x
    # and j-th sample in xtrain
    for i in range(N):
        for j in range(M):
            dist[i,j] = np.linalg.norm(x[i] - xtrain[j])
    return dist

# # %%

def getErrorRate(predictedRes, yActual):
    predictedRes = predictedRes.astype(int)
    yActual = yActual.flatten().astype(int)
    return np.mean(predictedRes != yActual)


# Load data from spamData.mat
xtrain, ytrain, xtest, ytest = du.loadData('spamData.mat')
# Log-transformation
xtrain = np.log(xtrain + 0.1)
xtest = np.log(xtest + 0.1)

distanceTrain = np.argsort(getDistance(xtrain, xtrain))
distanceTest = np.argsort(getDistance(xtrain, xtest))

K = getK()

trainErr = np.zeros(len(K))
testErr = np.zeros(len(K))

for i in range(len(K)):
    k = K[i]
    # Training Set
    # find the k-nearest neighbors and get index array
    kNearestNeighborIndexexTrain = distanceTrain[:, 0:k]
    # get the labels of k-nearest neighbors
    kLabelsTrain = np.squeeze(ytrain[kNearestNeighborIndexexTrain], axis = -1)

    # Test Set
    # find the k-nearest neighbors and get index array
    kNearestNeighborIndexexTest = distanceTest[:, 0:k]
    # get the labels of k-nearest neighbors
    kLabelsTest = np.squeeze(ytrain[kNearestNeighborIndexexTest], axis = -1)

    pTrain = np.zeros((2, len(xtrain)))
    pTest = np.zeros((2, len(xtest)))
    
    # compute the prob of being labelled as 0 and 1 respectively
    for j in [0,1]:
        pTrain[j] = np.sum(kLabelsTrain == j , axis = 1) / k
        pTest[j] = np.sum(kLabelsTest == j , axis = 1) / k

    # get the class labels
    predictedTrain = np.argmax(pTrain, axis = 0)
    predictedTest = np.argmax(pTest, axis = 0)
    
    # get error rate
    trainErr[i] = getErrorRate(predictedTrain, ytrain)
    testErr[i] = getErrorRate(predictedTest, ytest)



errRateTrain = getErrorRate(predictedTrain, ytrain)
errRateTrain = getErrorRate(predictedTest, ytest)



plt.plot(K, trainErr, "green", label="Training Set")
plt.plot(K, testErr, "red", label="Test Set")
plt.xlabel("k")
plt.ylabel("error rate")
plt.title("Q4. K-Nearest Neighbors")
plt.legend()
plt.show()


# %%
# Traing and testing error rates for k = 1, 10, 100
for i in [0, 9, len(K)-1]:
    print('K =', int(K[i]))
    print('training error:', trainErr[i])
    print('testing error:', testErr[i])