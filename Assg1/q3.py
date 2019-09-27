import scipy.io as sio
import numpy as np
import DataUtil as du
import math
import matplotlib.pyplot as plot
import scipy.stats as sst

np.seterr(divide='ignore')

# %%
# Load data from spamData.mat
xtrain, ytrain, xtest, ytest = du.loadData('spamData.mat')
# Log-transformation
xtrain = np.log(xtrain + 0.1)
xtest = np.log(xtest + 0.1)
