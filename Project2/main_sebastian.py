import autograd.numpy as np
from autograd import elementwise_grad
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from common_sebastian import (FrankeFunction, create_X, scale, model_terrain)
from methods_sebastian import OLS, Ridge

def main():
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

    n_epochs = 10000 # Number of epochs.
    M = 50   #size of each minibatch (10 gave good results)
    t0, t1 = 5, 50 # Paramters used in learning rate.
    gamma = 0.9 # Paramter used in momentum SGD.
    tol = 0.0001 # Tolerance for sum between analytical and numerical gradient.

    model = OLS(z, X, 0) # Initialize our OLS model.
    MSE1, beta1 = model.SGD(x, y, z_data, n_epochs, M, t0, t1, tol)
    MS2, beta2 = model.GDM(x, y, z_data, n_epochs, M, t0, t1, v, gamma, tol)
    # Plots for OLS:
    model_terrain(X, x, y, beta1, N, 'Stochastic gradient descent for OLS', z_data)
    model_terrain(X, x, y, beta2, N, 'Stochastic gradient descent with momentum for OLS', z_data)

    model = Ridge(z,X, 4.28*10**(-2)) # Initialize our Ridge model.
    MSE1, beta1 = model.SGD(x, y, z_data, n_epochs, M, t0, t1, tol)
    MS2, beta2 = model.GDM(x, y, z_data, n_epochs, M, t0, t1, v, gamma, tol)
    # Plots for Ridge:
    model_terrain(X, x, y, beta1, N, 'Stochastic gradient descent for Ridge', z_data)
    model_terrain(X, x, y, beta2, N, 'Stochastic gradient descent with momentum for Ridge', z_data)

main() 
