import scipy.io as sio
import numpy as np
from sklearn import preprocessing

mat = sio.loadmat('spamData.mat')

xtest = mat['Xtest']
xtrain = mat['Xtrain']
ytest = mat['ytest']
ytrain = mat['ytrain']

# Binarization preprocessing
binarizer = preprocessing.Binarizer().fit(xtest)
xtest = binarizer.transform(xtest)
print(xtest)

binarizer = preprocessing.Binarizer().fit(xtrain)
xtrain = binarizer.transform(xtrain)
print(xtrain)
