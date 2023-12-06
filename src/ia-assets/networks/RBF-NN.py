# -*- coding: utf-8 -*-
"""Red Neuronal de Base Radial.ipynb
#Red Neuronal de Base Radial
_Autor_: Felipe Jimenez
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.cluster import KMeans

"""### Clase RBF_NN"""

class RBF_NN:
    def __init__(self, h_hidden=15):
        self.nh = h_hidden

    def predict(self, X):
        G = np.exp(-(distance.cdist(X, self.C)) ** 2 / self.sigma ** 2)
        return G @ self.w 

    def predict_class(self, X):
        G = np.exp(-(distance.cdist(X, self.C)) ** 2 / self.sigma ** 2)
        return np.argmax(G @ self.w, axis=1) 

    def train(self, X, Y):
        self.ni, self.no = X.shape[1], Y.shape[1]
        km = KMeans(n_clusters=self.nh).fit(X)
        self.C = km.cluster_centers_
        self.sigma = (self.C.max() - self.C.min()) / (np.sqrt(2*self.nh))

        G = np.exp(-(distance.cdist(X, self.C)) ** 2 / self.sigma ** 2)
        self.w = np.linalg.pinv(G) @ Y

################################## Problem 1
# points = 500
# xl, xu = -5, 5
# x = np.linspace(xl, xu, points).reshape(-1, 1)
# y = 2 * np.cos(x) + np.sin(3 * x) + 5 + 0.2 * np.random.randn(*x.shape)

# neurons = 10
# net = RBF_NN(neurons)
# net.train(x, y)

# # draw
# plt.figure(figsize=(8, 6))
# plt.title('n = ' + str(neurons), fontsize=20)
# plt.plot(x, y, 'or', markersize=2)
# plt.plot(x, net.predict(x), '-b', markersize=2)


################################## Problem 2
x = np.genfromtxt("moons.csv", delimiter=',', skip_header=1, usecols=[0, 1])
y = np.genfromtxt("moons.csv", delimiter=',', skip_header=1, usecols=[2]).T.reshape(-1, 1)
y_tmp = 1-y
y = np.concatenate((y, y_tmp), axis=1)

neurons = 1000
net = RBF_NN(neurons)
net.train(x, y)

# draw
plt.figure(figsize=(6, 6))
plt.grid()
plt.title("n = " + str(neurons))

# # points draw
# for i in range(x.shape[0]):
#     if y[i][0] == 0:
#         plt.plot(x[i,0], x[i,1], 'ro', markersize=4)
#     else:
#         plt.plot(x[i,0], x[i,1], 'bo',markersize=4)

# points draw
y_est = net.predict_class(x)
for i in range(x.shape[0]):
    if y_est[i] == 1:
        plt.plot(x[i,0], x[i,1], 'ro', markersize=4)
    else:
        plt.plot(x[i,0], x[i,1], 'bo',markersize=4)

plt.show()
