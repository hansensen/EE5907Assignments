import scipy.io as sio
mat = sio.loadmat('spamData.mat')

xtest = mat['Xtest']
xtrain = mat['Xtrain']
ytest = mat['ytest']
ytrain = mat['ytrain']
