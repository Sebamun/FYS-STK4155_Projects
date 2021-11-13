import autograd.numpy as np
from autograd import elementwise_grad
from common_sebastian import (MSE, learning_schedule)
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.utils import shuffle

np.random.seed(1235)

class GradientDecent:
    def __init__(self, z, X, m, M):
        self.z = z # Data.
        self.X = X # Design matrix.
        self.M = M
        self.m = m

    def SGD(self, n_epochs, lmbd, gamma, v, t0, t1):
        # The stochastic gradient descent.
        w = np.random.randn(np.shape(self.X)[1],1) # Generate random initial w values.
        for epoch in range(n_epochs):
            self.X_, self.z_ = shuffle(self.X, self.z) # Shuffle the data for each epoch.
            for i in range(self.m):
                xi = self.X_[self.M*self.m : self.M*(self.m+1)]
                zi = self.z_[self.M*self.m : self.M*(self.m+1)]
                z_pred = xi@w # Our model predictio
                probability = self.activation_func(z_pred)
                gradient = self.der_crossEntropy(zi, probability, xi) + lmbd*w
                eta = learning_schedule(epoch*self.m+i, t0, t1) # Change the learning rate.

                # Stochastic gradient descent with momentum
                v = gamma*v + eta * gradient
                w = w - v


        return w

class LR(GradientDecent):
    def der_crossEntropy(self, y, y_o, x):
        val = x.T@(y_o - y)
        # val = np.mean(x.T@(y_o - y), axis = 1)
        return(val.reshape(-1,1))

    def activation_func(self, x): # Sigmoid
        return 1/(1 + np.exp(-x))

