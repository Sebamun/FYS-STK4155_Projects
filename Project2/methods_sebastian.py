import autograd.numpy as np
from autograd import elementwise_grad
from common_sebastian import (MSE, learning_schedule)
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge

class GradientDecent:
    def __init__(self, z, X, m, M, lamb):
        self.z = z # Data.
        self.X = X # Design matrix.
        self.lamb = lamb # Parameter used in ridge.
        self.M = M
        self.m = m

    def SGD(self, x, y, z_data, n_epochs, M, t0, t1, tol):
        # The stochastic gradient descent.
        m = int(len(self.X)/M) #number of minibatches
        beta = np.random.randn(np.shape(self.X)[1],1) # Generate random initial beta values.
        for epoch in range(n_epochs):
            for i in range(m):
                gradients = elementwise_grad(self.cost_func) # Use autograd to calculate the gradient of the cost function.
                eta = learning_schedule(epoch*m+i,t0,t1) # Change the learning rate.
                # Stochastic gradient descent:
                beta = beta - eta * gradients(beta)
                if np.linalg.norm(gradients(beta))<0.01: # If less than tolarance we stop our gradient descent.
                        break

        z_pred = self.X@beta # Our model prediction.
        mean_squared_error_1 = MSE(self.z, z_pred) # Collect the mean square error.
        # Check for autograd function for gradient:
        gradient_num = elementwise_grad(self.gradient_numerical)
        assert np.all(np.squeeze(self.gradient_analytical(beta))[:] \
        - np.squeeze(gradient_num(beta))[:] < tol)

        return mean_squared_error_1, beta


    def GDM(self, x, y, z_data, n_epochs, M, t0, t1, v, gamma, tol):
        # The momentum stochastic gradient descent.
        m = int(len(self.X)/M) #number of minibatches
        beta = np.random.randn(np.shape(self.X)[1],1) # Generate random initial beta values.
        for epoch in range(n_epochs):
            for i in range(m):
                gradients = elementwise_grad(self.cost_func) # Use autograd to calculate the gradient of the cost function.
                eta = learning_schedule(epoch*m+i,t0,t1) # Change the learning rate.
                # Stochastic gradient descent with momentum:
                v = gamma*v + eta*gradients(beta)
                beta = beta - v
                if np.linalg.norm(gradients(beta))<0.01: # If less than tolarance we stop our gradient descent.
                    break

        z_pred = self.X@beta # Our model prediction.
        mean_squared_error_2 = MSE(self.z, z_pred) # Collect the mean square error.
        # Check for autograd function for gradient:
        gradient_num = elementwise_grad(self.gradient_numerical)
        assert np.all(np.squeeze(self.gradient_analytical(beta))[:] \
        - np.squeeze(gradient_num(beta))[:] < tol)

        return mean_squared_error_2, beta

    def compare_MSE(self, n_epochs, t0, eta0):
        sgdreg = SGDRegressor(loss='squared_loss', max_iter = n_epochs, penalty = None , eta0 = eta0)
        #sgdreg = SGDRegressor(loss = 'squared_loss', max_iter = n_epochs, penalty=None, alpha = 1/t0, learning_rate='optimal')
        sgdreg.fit(self.X, self.z.ravel())
        beta_scikit = sgdreg.coef_
        z_pred = self.X@beta_scikit
        MSE_sci = MSE(self.z, z_pred)

        beta = np.random.randn(np.shape(self.X)[1],1) # Generate random initial beta values.
        for epoch in range(n_epochs):
            for i in range(self.m):
                gradients = elementwise_grad(self.cost_func) # Use autograd to calculate the gradient of the cost function.
                eta = eta0 # Change the learning rate.
                # Stochastic gradient descent:
                beta = beta - eta * gradients(beta)
                #if np.linalg.norm(gradients(beta))<0.01: # If less than tolarance we stop our gradient descent.
                        #break

        z_pred = self.X@beta # Our model prediction.
        MSE_own = MSE(self.z, z_pred) # Collect the mean square error.

        # Write to file:
        f = open("Textfiles/MSE_comparison.txt", "w")
        f.write('Mean squared error from scikit and our model for OLS\n')
        f.write(f' Scikit SGD: {MSE_sci}\n')
        f.write(f' Own SGD: {MSE_own}\n')
        f.close()

class OLS(GradientDecent):
    def cost_func(self, beta):
        # The cost function for OLS.
        random_index = np.random.randint(self.m) # Generate random index used in distribution of batches.
        xi = self.X[random_index*self.M:(random_index+1)*self.M]
        zi = self.z[random_index*self.M:(random_index+1)*self.M]
        return 1/(len(xi)) * np.sum( (zi - (xi @ beta))**2 )

    def gradient_numerical(self, beta):
        # Used to compare with analytical expression.
        return 1/(len(self.X)) * np.sum( (self.z - (self.X @ beta))**2 )

    def gradient_analytical(self, beta):
        # Analytical expression for gradient of cost function for OLS.
        return 2.0/len(self.X)*self.X.T @ ((self.X @ beta)-self.z)

class Ridge(GradientDecent):
    def cost_func(self, beta):
        random_index = np.random.randint(self.m) # Generate random index used in distribution of batches.
        xi = self.X[random_index*self.M:(random_index+1)*self.M]
        zi = self.z[random_index*self.M:(random_index+1)*self.M]
        # The cost function for Ridge.
        return 1/ len(xi) * np.sum( (zi - (xi @ beta))**2 ) + np.sum(beta.T@beta) * self.lamb

    def gradient_numerical(self, beta):
        # Used to compare with analytical expression.
        return 1/ len(self.X) * np.sum( (self.z - (self.X @ beta))**2 ) + np.sum(beta.T@beta) * self.lamb

    def gradient_analytical(self, beta):
        # Analytical expression for gradient of cost function for Ridge.
        return (2.0/len(self.X)*self.X.T @ (self.X @ (beta)-self.z)+2*self.lamb*beta)
