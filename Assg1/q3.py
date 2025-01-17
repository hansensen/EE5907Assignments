
# %%
import scipy.io as sio
import numpy as np
import DataUtil as du
import matplotlib.pyplot as plot
import scipy.stats as sst
import matplotlib.pyplot as plt
np.seterr(divide='ignore')


# %%

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# get an array of lambda
def getLambda():
    lambda1 = np.arange(1, 11, 1)
    lambda2 = np.arange(15, 105, 5)
    return np.concatenate((lambda1, lambda2), axis=0)


# get gradient of NLL(wBold) without regularisation term
def getG(xtrainBias, wBold):
    mu = sigmoid(xtrainBias.dot(wBold))
    G = xtrainBias.transpose().dot(mu - ytrain)
    return G


# get Hession of NLL(wBold) without regularisation term
def getH(xtrainBias, wBold):
    # S = N x N diagonal matrix, where i-th diagonal is mu_i(1 - mu_i)
    mu = sigmoid(xtrainBias.dot(wBold))
    S = np.diag(np.squeeze((mu * (1 - mu)), axis=1))
    # H = Xt x S x X
    H = xtrainBias.transpose().dot(S)
    H = H.dot(xtrainBias)
    return H


# %%
# Newton's Method, get w to minimise NLL
def newtonsMethod(xtestBias, xtrainBias, lam):
    # Get dimensions
    # D: # of features, xtrainBias has D + 1 columns due to bias term
    D = len(xtrainBias[0]) - 1
    # Initialisation, wBold is a (D+1) x 1 vector
    wBold = np.zeros((D + 1, 1))
    diff = 999
    # Deploy Newton's method to obtain optimal w
    while (diff > 0.01):
        # get gradient of NLLreg(wBold)
        gReg = getG(xtrainBias, wBold) + lam * \
            np.concatenate((np.zeros((1, 1)), wBold[1:]), axis=0)
        # get hession of NLLreg(wBold)
        # I is a (D+1)x(D+1) indentity matrix except top left corner is 0 instead of 1
        I = np.identity(D + 1)
        I[0, 0] = 0
        hReg = getH(xtrainBias, wBold) + lam * I

        # Solve Hk * dk = -gk for dk
        dk = ((np.linalg.pinv(hReg)).dot(-gReg))

        # No need to use line search as lecturer mentioned during the lecture :)
        wBold += dk

        diff = np.sum(np.abs(dk))
    return wBold


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

lambdaArr = getLambda()

trainErr = np.zeros(len(lambdaArr))
testErr = np.zeros(len(lambdaArr))

for i in range(len(lambdaArr)):
    lam = lambdaArr[i]
    wBold = newtonsMethod(xtestBias, xtrainBias, lam)
    # Training Set
    pSpam = sigmoid(xtrainBias.dot(wBold))
    trainErr[i] = (1 - np.sum((pSpam >= 0.5) == ytrain) / len(ytrain))

    # Test Set
    pSpam = sigmoid(xtestBias.dot(wBold))
    testErr[i] = (1 - np.sum((pSpam >= 0.5) == ytest) / len(ytest))

# %%
# Traing and testing error rates for alpha = 1, 10, 100
for i in [0, 9, len(lambdaArr)-1]:
    print('lambda =', int(lambdaArr[i]))
    print('training error:', trainErr[i])
    print('testing error:', testErr[i])


# %%
plt.plot(lambdaArr, trainErr, "green", label="Training Set")
plt.plot(lambdaArr, testErr, "red", label="Test Set")
plt.xlabel("lambda")
plt.ylabel("error rate")
plt.title("Q3. Logistic Regression")
plt.legend()
plt.show()


# %%
