import numpy as np
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.utils import shuffle

from common import (MSE, learning_schedule)

np.random.seed(1235)

def der_crossEntropy(y, y_o, x):
    val = x.T@(y_o - y)
    return(val.reshape(-1,1))

class GradientDecent:
    def __init__(self, z, X, m, M):
        self.z = z  # Data.
        self.X = X  # Design matrix.
        self.M = M
        self.m = m

    def SGD(self, n_epochs, lmbd, gamma, v, t0, t1):
        w = np.random.randn(np.shape(self.X)[1],1)              # Generate random initial w values.
        for epoch in range(n_epochs):
            self.X_, self.z_ = shuffle(self.X, self.z)          # Shuffle the data for each epoch.
            for i in range(self.m):
                xi = self.X_[self.M*self.m : self.M*(self.m+1)]
                zi = self.z_[self.M*self.m : self.M*(self.m+1)]
                z_pred = xi@w                                   # Our model predictio
                probability = self.activation_func(z_pred)
                gradient = der_crossEntropy(zi, probability, xi) + lmbd*w
                eta = learning_schedule(epoch*self.m+i, t0, t1) # Change the learning rate.

                # Adding momentum to our method
                v = gamma*v + eta * gradient
                w = w - v

        return w

class Sigmoid(GradientDecent):
    def activation_func(self, z):
        return 1/(1 + np.exp(-z))

