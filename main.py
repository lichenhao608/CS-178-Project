import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('semeion.data')
x = data[:,:256].reshape((-1,1,16,16))
y = data[:,256:]
