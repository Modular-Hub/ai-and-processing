import numpy as np
from sklearn import preprocessing

class LogisticNeuron:
    def __init__(self, n_inputs, learning_rate=0.1) -> None:
        self.w = -1 + 2 * np.random.rand(n_inputs)
        self.b = -1 + 2 * np.random.rand()
        self.eta = learning_rate

    def predict_proba(self, X):
        Z = np.dot(self.w, X) + self.b
        return (1 / (1 + np.exp(-Z)))

    def predict(self, X, umbral=0.5) -> int:
        Z = np.dot(self.w, X) + self.b
        Y_est = (1 / (1 + np.exp(-Z)))
        return 1.0 * (Y_est > umbral)

    def fit(self, X, Y, epochs=500):
        p = X.shape[1]
        for _ in range(epochs):
            Y_est = self.predict_proba(X)
            self.w += (self.eta/p) * np.dot((Y - Y_est), X.T).ravel()
            self.b += (self.eta/p) * np.sum(Y - Y_est)

if __name__=="__main__":
    problem = 'diabetes' # ['cancer', 'diabetes']

    if problem == 'cancer':
        epochs, learning_rate, I = 1000, 0.015, 9
    else:
        epochs, learning_rate, I = 10000, 0.1, 8

    X = np.genfromtxt(problem+".csv", delimiter=',', skip_header=1, usecols= [i for i in range(I)]).T
    Y = np.genfromtxt(problem+".csv", delimiter=',', skip_header=1, usecols= [I]).T

    for i in range(I):
        X[i, :] = preprocessing.minmax_scale(X[i, :])

    neuron = LogisticNeuron(X.shape[0], learning_rate)
    neuron.fit(X, Y, epochs=epochs)

    predic = neuron.predict(X)
    good = 0
    for i in range(X.shape[1]):
        if ( predic[i] == Y[i] ):
            good += 1
