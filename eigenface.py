# Failed I think, 10% accuracy only

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.preprocessing import StandardScaler

data = np.loadtxt('semeion.data')
x = data[:, :256]
imgs = x.reshape((-1, 1, 16, 16))
y = data[:, 256:]
y = np.where(y == 1)[1].reshape((-1, 1))

'''
mu = np.mean(x, axis=0)
x0 = x-mu
'''
xtrain = x[1::2]
xtest = x[::2]
ytrain = y[1::2]
ytest = y[::2]

xtrain_mean = np.mean(xtrain, axis=0)
trainX = xtrain - xtrain_mean

numClasses = 10
numEig = 16*16
picSize = 16*16

'''
U, s, Vh = linalg.svd(x0, full_matrices=False)
W = np.dot(U, np.diag(s))
'''

cov = np.cov(trainX.T)
w, Vh = np.linalg.eig(cov)
ws = np.sort(w)
ws = ws[::-1]
for i in range(0, numEig):
    Vh[:, i] = Vh[:, np.where(w == ws[i])[0][0]]

Vh = Vh[:, :numEig].real

del x, cov, w, ws

omega = np.zeros((numClasses, numEig, picSize))
for i in range(numClasses):
    trainD = trainX[np.where(ytrain == i)]
    for k in range(len(trainD)):
        temp = Vh.T*trainD[k]
        omega[i] += temp
    omega[i] /= len(trainD)


correct = 0
for k in range(len(xtest)):
    omega_m = Vh.T*(xtest[i]-xtrain_mean)
    dist = np.zeros((numClasses))
    for i in range(0, numClasses):
        dist[i] = np.linalg.norm(omega[i] - omega_m)

    if int(dist.argmin()) == y[k, 0]:
        correct += 1
print(correct/len(xtest))
