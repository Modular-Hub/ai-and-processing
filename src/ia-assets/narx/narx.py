import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

"""## Delay Block"""

class DelayBlock:
    def __init__(self, n_inputs, n_delays):
        self.memory = np.zeros((n_inputs, n_delays))

    def add(self, X):
        x_new = X.copy().reshape(1, -1)
        self.memory[:, 1:] = self.memory[:, :-1]
        self.memory[:, 0] = x_new
    
    def get(self):
        return self.memory.reshape(-1, 1, order='F')
    
    def add_and_get(self, X):
        self.add(X)
        return self.memory.reshape(-1, 1, order='F')

"""## MLP"""

def linear(z, derivative=False):
    a = z
    if derivative:
        da = np.ones(z.shape)
        return a, da

    return a

def logistic(z, derivative=False):
    a = 1 / (1 + np.exp(-z))

    if derivative:
        da = np.ones(z.shape)
        return a, da

    return a

def logistic_hidden(z, derivative=False):
    a = 1 / (1 + np.exp(-z))

    if derivative:
        da = a * (1 - a)
        return a, da

    return a

def tanh(z, derivative=False):
    a = np.tanh(z)
    if derivative:
        da = (1 + a) * (1 - a)
        return a, da

    return a

def relu(z, derivative=False):
    a = z * (z >= 0)
    if derivative:
        da = np.array(z >= 0, dtype=float)
        return a, da

    return a

def softMax(z, derivative=False):
    e_z = np.exp(z - np.max(z, axis=0))
    a = e_z / np.sum(e_z, axis=0)
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a

class MLP:
    def __init__(self, layers_dim,
        hidden_activation=tanh,
        output_activation=logistic):

        # Attributes
        self.L = len(layers_dim) - 1
        self.w = [None] * (self.L + 1)
        self.b = [None] * (self.L + 1)
        self.f = [None] * (self.L + 1)

        # initialize weighs adn biases
        for l in range(1, self.L + 1):
            self.w[l] = -1 + 2 * np.random.rand(layers_dim[l], layers_dim[l-1])
            self.b[l] = -1 + 2 * np.random.rand(layers_dim[l], 1)

            if l == self.L:
                self.f[l] = output_activation
            else:
                self.f[l] = hidden_activation

    def predict(self, X):
        a = X
        for l in range(1, self.L + 1):
            z = self.w[l] @ a + self.b[l]
            a = self.f[l](z)

        return a

    def train(self, X, Y, epochs=500, lr=0.1):
        p = X.shape[1]

        for _ in range(epochs):
            # Initialize activations and gradients
            a = [None] * (self.L + 1)
            da = [None] * (self.L + 1)
            lg = [None] * (self.L + 1)

            # Propagation
            a[0] = X
            for l in range(1, self.L + 1):
                z = self.w[l] @ a[l-1] + self.b[l]
                a[l], da[l] = self.f[l](z, derivative=True)

            # Backpropagation
            for l in range(self.L, 0, -1):
                if l == self.L:
                    lg[l] = -(Y - a[l]) * da[l]
                else:
                    lg[l] = (self.w[l+1].T @ lg[l+1]) * da[l]

            # Gradient Descent
            for l in range(1, self.L + 1):
                self.w[l] -= (lr/p) * (lg[l] @ a[l-1].T)
                self.b[l] -= (lr/p) * np.sum(lg[l])

"""## NARX"""

class NARX:
  def __init__ (self, n_inputs, n_outputs, n_delays, dense_hidden_layers=(100,), learning_rate=0.01, n_repeat_train=5):
    self.net = MLP (((n_inputs+n_outputs)*n_delays, *dense_hidden_layers, n_outputs), output_activation=linear)
    self.dbx = DelayBlock(n_inputs, n_delays)
    self.dby = DelayBlock(n_outputs, n_delays)
    self.learning_rate = learning_rate
    self.n_repeat_train = n_repeat_train

  def predict(self, x):
    X_block = self.dbx.add_and_get(x)
    Y_est_block = self.dby.get()
    net_input = np.vstack((X_block, Y_est_block))
    # Neural Network Prediction
    y_est = self.net.predict(net_input)

    # Save recurrent block prediction
    self.dby.add(y_est)
    return y_est

  def predict_and_train(self, x, y):
    X_block = self.dbx.add_and_get(x)
    Y_est_block = self.dby.get()
    net_input = np.vstack((X_block, Y_est_block))
    # Neural Network Prediction
    y_est = self.net.predict(net_input)
    self.net.train(net_input, y, epochs=self.n_repeat_train, lr=self.learning_rate)
    
    # Save recurrent block prediction
    self.dby.add(y_est)
    return y_est

"""## Test"""

df = pd.read_csv("daily-min-temperatures.csv", skiprows=1, usecols=[1])
da = np.asarray(df)

# Test NARX a un paso
narx = NARX(1, 1, 10,
            dense_hidden_layers=(100,),
            learning_rate=0.01, n_repeat_train=1)
y_est = np.zeros((1, da.shape[0]))

for i in range(data_array.shape[0]-1):
    x = da[i]
    y = da[i+1]
    y_est[0, i] = narx.predict_and_train(x, y)

#Gráfica
plt.figure(figsize=(13, 10))
plt.plot(da, lw=0.7)
plt.plot(y_est[0], lw=0.2)
plt.xlabel("Días")
plt.ylabel("Temperatura")
plt.title("Predictor a un paso NARX")
plt.show()