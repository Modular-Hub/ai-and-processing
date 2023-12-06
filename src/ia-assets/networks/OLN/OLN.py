import numpy as np
import matplotlib.pyplot as plt

def linear(z, derivative=False):
    a = z
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a

def logistic(z, derivative=False):
    a = 1 / (1 + np.exp(-z))

    if derivative:
        da = np.ones(z.size)
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
    
"""
    One-Layer Network
    Felipe Jimenez
"""
class OLN:
    def __init__(self, n_inputs, n_outputs, activation_f=linear) -> None:
        self.w = -1 + 2 * np.random.rand(n_outputs, n_inputs)
        self.b = -1 + 2 * np.random.rand(n_outputs, 1)
        self.f = activation_f
    
    def predict(self, X):
        Z = self.w @ X + self.b
        return self.f(Z)


    def fit(self, X, Y, epochs=500, lr=0.1) -> None:
        p = X.shape[1]
        for _ in range(epochs):
            # Propagation
            Z = self.w @ X + self.b
            Yest, dY = self.f(Z, derivative=True)

            # Calculate Local Gradient 
            lg = (Y - Yest) * dY

            # Update
            self.w += (lr/p) * lg @ X.T
            self.b += (lr/p) * np.sum(lg)
