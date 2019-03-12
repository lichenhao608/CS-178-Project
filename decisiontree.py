# Yeah, we done

import numpy as np
from sklearn.tree import DecisionTreeClassifier

data = np.loadtxt('semeion.data')
x = data[:, :256]
imgs = x.reshape((-1, 1, 16, 16))
y = data[:, 256:]
y = np.where(y == 1)[1]

trainX = x[:1000]
trainY = y[:1000]

testX = x[1000:]
testY = y[1000:]

clf = DecisionTreeClassifier()
clf.fit(trainX, trainY)

p = clf.predict(testX)

print(p)
print(np.sum(p == testY)/len(testY))
