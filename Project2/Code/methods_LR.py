import autograd.numpy as np
from autograd import elementwise_grad
from common_sebastian import (MSE, learning_schedule)
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
import time
from sklearn.utils import shuffle

np.random.seed(1235)

class GradientDecent:
    def __init__(self, z, X, m, M, lmbd):
        self.z = z # Data.
        self.X = X # Design matrix.
        self.M = M
        self.m = m
        self.lmbd = lmbd

    def SGD(self, n_epochs, t0, t1, timer):
        # The stochastic gradient descent.
        start = time.time() # Start timer.
        beta = np.random.randn(np.shape(self.X)[1],1) # Generate random initial beta values.
        for epoch in range(n_epochs):
            self.X_, self.z_ = shuffle(self.X, self.z) # Shuffle the data for each epoch.
            for i in range(self.m):
                xi = self.X_[self.M*self.m : self.M*(self.m+1)]
                zi = self.z_[self.M*self.m : self.M*(self.m+1)]
                z_pred = xi@beta # Our model predictio
                probability = self.activation_func(z_pred)
                gradient = self.der_crossEntropy(zi, probability, xi) + self.lmbd*beta
                eta = learning_schedule(epoch*self.m+i, t0, t1) # Change the learning rate.

                # Stochastic gradient descent:
                beta = beta - eta * gradient


        # train_pred = self.activation_func(z_pred)

        end = time.time() # End timer.
        if timer == True:
            f = open("Textfiles/time.txt", "a")
            f.close()

        return gradient, beta

class OLS(GradientDecent):
    def der_crossEntropy(self, y, y_o, x):
        val = x.T@(y_o - y)
        # val = np.mean(x.T@(y_o - y), axis = 1)
        return(val.reshape(-1,1))

    def activation_func(self, x): # Sigmoid
        return 1/(1 + np.exp(-x))

