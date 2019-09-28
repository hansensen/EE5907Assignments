
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

# %%
# Load data from spamData.mat
xtrain, ytrain, xtest, ytest = du.loadData('spamData.mat')
# Log-transformation
xtrain = np.log(xtrain + 0.1)
xtest = np.log(xtest + 0.1)
# Add bias term 1 to the start of x
numSample = xtrain.shape[0]
xtrainBias = np.concatenate((np.ones((numSample, 1)), xtrain), axis=1)
numSample = xtest.shape[0]
xtestBias = np.concatenate((np.ones((numSample, 1)), xtest), axis=1)

K = getK()

trainErr = np.zeros(len(K))
testErr = np.zeros(len(K))

for i in range(len(K)):
    k = K[i]
    # Training Set

    # Test Set

# %%
# Traing and testing error rates for alpha = 1, 10, 100
for i in [0, 9, len(K)-1]:
    print('lambda =', int(K[i]))
    print('training error:', trainErr[i])
    print('testing error:', testErr[i])


# %%
plt.plot(K, trainErr, "green", label="Training Set")
plt.plot(K, testErr, "red", label="Test Set")
plt.xlabel("k")
plt.ylabel("error rate")
plt.title("Q4. K-Nearest Neighbors")
plt.legend()
plt.show()


# %%
