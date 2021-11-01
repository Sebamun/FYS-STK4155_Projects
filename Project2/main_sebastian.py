import autograd.numpy as np
from autograd import elementwise_grad
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from common_sebastian import (FrankeFunction, create_X, scale)
from methods_sebastian import OLS, Ridge
from plot_sebastian import model_terrain, MSE_lamb

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

    n_epochs = 2000 # Number of epochs. 10000
    M = 10   #size of each minibatch (10 gave good results)
    t0, t1 = 5, 50 # Paramters used in learning rate.
    gamma = 0.9 # Paramter used in momentum SGD.
    tol = 0.0001 # Tolerance for sum between analytical and numerical gradient.
    v = 0 # Initial velocity for stochastic gradient descent with momentum.

    lamb = 0
    model = OLS(z, X, lamb) # Initialize our OLS model.
    MSE1_OLS, beta1 = model.SGD(x, y, z_data, n_epochs, M, t0, t1, tol)
    MSE2_OLS, beta2 = model.GDM(x, y, z_data, n_epochs, M, t0, t1, v, gamma, tol)
    # Plots for OLS:
    model_terrain(X, x, y, beta1, N, 'Stochastic gradient descent for OLS', z_data)
    model_terrain(X, x, y, beta2, N, 'Stochastic gradient descent with momentum for OLS', z_data)
    lamb =[1*10**(-4), 2.07*10**(-3), 4.28*10**(-2), 3*10**(-1), 1] #np.linspace(1*10**(-4), 1, 10) #[1*10**(-4), 2.07*10**(-3), 4.28*10**(-2), 10]
    # Plots for Ridge:
    MSE1_Ridge_plot = np.zeros(len(lamb)) # Collect the MSE after n_epochs iterations.
    MSE2_Ridge_plot = MSE1_Ridge_plot
    for l in range(len(lamb)): # Run through our lambda values.
        model = Ridge(z, X, lamb[l]) # Initialize our Ridge model.
        MSE1_Ridge, beta1 = model.SGD(x, y, z_data, n_epochs, M, t0, t1, tol)
        MSE2_Ridge, beta2 = model.GDM(x, y, z_data, n_epochs, M, t0, t1, v, gamma, tol)
        MSE1_Ridge_plot[l] = MSE1_Ridge[-1]
        MSE2_Ridge_plot[l] = MSE2_Ridge[-1]

        # Plots for Ridge:
        model_terrain(X, x, y, beta1, N, 'Stochastic gradient descent for Ridge', z_data)
        model_terrain(X, x, y, beta2, N, 'Stochastic gradient descent with momentum for Ridge', z_data)
    MSE_lamb(MSE1_Ridge_plot, MSE2_Ridge_plot, lamb)
    plt.show()


main()
