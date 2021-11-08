import autograd.numpy as np
from autograd import elementwise_grad
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from common_sebastian import (FrankeFunction, create_X, scale, MSE)
from methods_sebastian import OLS, Ridge
from plot_sebastian import model_terrain, MSE_lamb

def main_1():
    np.random.seed(2018) # Generate random values from seed.

    N = 10 # Number of datapoints.
    polydegree = 5 # Degree for polynomial.
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y)
    z_data = z # Copy of the real data.
    z = np.ravel(z)[:, np.newaxis] # Removes one dimension and add axis to each element.
    X = create_X(x, y, polydegree) # Creates design matix.
    X = scale(X) # Scale our matrix.

    n_epochs = 1000 # Number of epochs. 10000
    M = 10   #size of each minibatch (10 gave good results)
    t0, t1 = 0.5, 100 # Paramters used in learning rate. # 50
    gamma = 0.9 # Paramter used in momentum SGD.
    tol = 0.0001 # Tolerance for sum between analytical and numerical gradient.
    v = 0 # Initial velocity for stochastic gradient descent with momentum.
    m = int(len(X)/M) # Used when we split into minibatches.
    alph_1= 0.5 # The alph values are used for which transparency to use in plotting of model and data.
    alph_2 = 1.0
    optimal_lambda = 4.28*10**(-2) # found this in last project.

    model = OLS(z, X, m, M, 0) # Initialize our OLS model.
    # Calculate methods for OLS:
    MSE1_OLS, beta1 = model.SGD(x, y, z_data, n_epochs, t0, t1, tol, True)
    MSE2_OLS, beta2 = model.GDM(x, y, z_data, n_epochs, t0, t1, v, gamma, tol, True)
    # Plots for OLS:
    model_terrain(X, x, y, beta1, N, 'Stochastic gradient descent for OLS', z_data, alph_2, alph_1)
    model_terrain(X, x, y, beta2, N, 'Stochastic gradient descent with momentum for OLS', z_data, alph_1, alph_2)
    # Compare with scikit: (we only have loss model for OLS).
    eta0 = 0.001 # When we compare with scikitlearn we use a constant learning rate.
    model.compare_MSE(n_epochs, t0, eta0)

    model = Ridge(z, X, m, M, optimal_lambda) # Initialize our Ridge model with specific lambda value.
    # This was the optimal lambda value from the last project.
    # Calculate methods for Ridge:
    MSE1_Ridge, beta1 = model.SGD(x, y, z_data, n_epochs, t0, t1, tol, False)
    MSE2_Ridge, beta2 = model.GDM(x, y, z_data, n_epochs, t0, t1, v, gamma, tol, False)
    # Plots for Ridge:
    model_terrain(X, x, y, beta1, N, 'Stochastic gradient descent for Ridge', z_data, alph_2, alph_1)
    model_terrain(X, x, y, beta2, N, 'Stochastic gradient descent with momentum for Ridge', z_data, alph_2, alph_1)

    return MSE2_Ridge


def main_2(optimal_MSE):

    np.random.seed(2018) # Generate random values from seed.

    N = 10 # Number of datapoints.
    polydegree = 5 # Degree for polynomial.
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y)
    z_data = z # Copy of the real data.
    z = np.ravel(z)[:, np.newaxis] # Removes one dimension and add axis to each element.
    X = create_X(x, y, polydegree) # Creates design matix.
    X = scale(X) # Scale our matrix.

    n_epochs = 1000 # Number of epochs. 10000
    M = 10   #size of each minibatch (10 gave good results)
    t0, t1 = 0.5, 100 # Paramters used in learning rate. # 50
    gamma = 0.9 # Paramter used in momentum SGD.
    tol = 0.0001 # Tolerance for sum between analytical and numerical gradient.
    v = 0 # Initial velocity for stochastic gradient descent with momentum.
    m = int(len(X)/M) # Used when we split into minibatches.
    optimal_lambda = 4.28*10**(-2)

    N_lambdas = 10 # The number of lambdas.
    lambdas = np.logspace(-4, 1, N_lambdas) # Generate different lambda values.
    lambdas[5] = optimal_lambda # We insert the optimal lambda value we found in the last project.

    MSE_Ridge_plot = np.zeros(len(lambdas)) # Collect the MSE after n_epochs iterations.
    for l in range(len(lambdas)): # Run through our lambda values.
        model_ = Ridge(z, X ,m ,M, lambdas[l]) # Initialize our Ridge model.
        MSE_Ridge, beta2_ = model_.GDM(x, y, z_data, n_epochs, t0, t1, v, gamma, tol, False)
        MSE_Ridge_plot[l] = MSE_Ridge # Collect MSE at last n for momentum SGD.

    ind = np.argmin(MSE_Ridge_plot)
    MSE_lamb(MSE_Ridge_plot, lambdas, ind, optimal_lambda)

    return lambdas[ind]


optimal_MSE = main_1()
main_2(optimal_MSE)
