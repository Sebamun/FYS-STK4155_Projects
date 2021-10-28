import autograd.numpy as np
from autograd import elementwise_grad
from common_sebastian import (MSE, learning_schedule)

class GradientDecent:
    def __init__(self, z, X, lamb):
        self.z = z # Data.
        self.X = X # Design matrix.
        self.lamb = lamb # Parameter used in ridge.

    def SGD(self, x, y, z_data, n_epochs, M, t0, t1, tol):
        # The stochastic gradient descent.
        m = int(len(self.X)/M) #number of minibatches
        mean_squared_error_1 = np.zeros(n_epochs) # Here we store our MSE values.
        beta = np.random.randn(np.shape(self.X)[1],1) # Generate random initial beta values.

        for epoch in range(n_epochs):
            for i in range(m):
                random_index = np.random.randint(m) # Generate random index used in distribution of batches.
                # Splitting our data into batches:
                xi = self.X[random_index*M:(random_index+1)*M]
                zi = self.z[random_index*M:(random_index+1)*M]
                gradients = elementwise_grad(self.cost_func) # Use autograd to calculate the gradient of the cost function.
                eta = learning_schedule(epoch*m+i,t0,t1) # Change the learning rate.
                # Stochastic gradient descent:
                beta = beta - eta*gradients(beta)
                z_pred = self.X@beta # Our model prediction.
                mean_squared_error_1[epoch] = MSE(self.z, z_pred) # Collect the mean square error.
                if np.linalg.norm(gradients(beta))<0.01: # If less than tolarance we stop our gradient descent.
                    break

        assert np.all(np.squeeze(self.gradient_analytical(beta))[:] \
        - np.squeeze(gradients(beta))[:] < tol)

        return mean_squared_error_1, beta

    def GDM(self, x, y, z_data, n_epochs, M, t0, t1, v, gamma, tol):
        # The momentum stochastic gradient descent.
        m = int(len(self.X)/M) #number of minibatches
        v = 0 # Initial velocity for stochastic gradient descent with momentum.
        mean_squared_error_2 = np.zeros(n_epochs) # Here we store our MSE values.
        beta = np.random.randn(np.shape(self.X)[1],1) # Generate random initial beta values.


        for epoch in range(n_epochs):
            for i in range(m):
                random_index = np.random.randint(m) # Generate random index used in distribution of batches.
                # Splitting our data into batches:
                xi = self.X[random_index*M:(random_index+1)*M]
                zi = self.z[random_index*M:(random_index+1)*M]
                gradients = elementwise_grad(self.cost_func) # Use autograd to calculate the gradient of the cost function.
                eta = learning_schedule(epoch*m+i,t0,t1) # Change the learning rate.
                # Stochastic gradient descent with momentum:
                v = gamma*v - eta*gradients(beta)
                beta = beta + v
                z_pred = self.X@beta # Our model prediction.
                mean_squared_error_2[epoch] = MSE(self.z, z_pred) # Collect the mean square error.
                if np.linalg.norm(gradients(beta))<0.01: # If less than tolarance we stop our gradient descent.
                    break

        # Check for autograd function for last gradient:
        assert np.all(np.squeeze(self.gradient_analytical(beta))[:] \
        - np.squeeze(gradients(beta))[:] < tol)

        return mean_squared_error_2, beta

class OLS(GradientDecent):
    def cost_func(self, beta):
        # The cost function for OLS.
        return 1/(len(self.X)) * np.sum( (self.z - (self.X @ beta))**2 )
    def gradient_analytical(self, beta):
        # Analytical expression for gradient of cost function for OLS.
        return 2.0/len(self.X)*self.X.T @ ((self.X @ beta)-self.z)

class Ridge(GradientDecent):
    def cost_func(self, beta):
        # The cost function for Ridge.
        return 1/ len(self.X) * np.sum( (self.z - (self.X @ beta))**2 ) + np.sum(beta.T@beta) * self.lamb
    def gradient_analytical(self, beta):
        # Analytical expression for gradient of cost function for Ridge.
        return (2.0/len(self.X)*self.X.T @ (self.X @ (beta)-self.z)+2*self.lamb*beta)
