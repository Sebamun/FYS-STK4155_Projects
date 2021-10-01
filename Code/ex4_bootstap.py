from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import sklearn.linear_model as skl

from common import FrankeFunction, create_X, MSE, scale

# For nice plots
plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

np.random.seed(2018)

N = 100
N_boostraps = 100
poly_degrees = np.arange(1, 10)

# Making meshgrid of datapoints and compute Franke's function
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y)

z = np.ravel(z)

nlambdas = 20
lambdas = np.logspace(-2, 1, nlambdas)

# MSE_ridge = np.zeros((2, len(poly_degrees), N_boostraps, nlambdas))
error = np.zeros((2, len(poly_degrees), nlambdas))
bias = np.zeros_like(error)
variance = np.zeros_like(error)

for idx, degree in enumerate(poly_degrees):
    # Repeat now for Ridge regression and various values of the regularization parameter
    X = create_X(x, y, n=degree)

    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    I = np.eye(X_train.shape[1], X_test.shape[1]) #Return a 2-D array with ones on the diagonal and zeros elsewhere

    X_train, X_test = scale(X_train, X_test)

    # The following (m x n_bootstraps) matrix holds the column vectors z_pred
    # for each bootstrap iteration.
    z_pred_train = np.empty((z_train.shape[0], N_boostraps, nlambdas))
    z_pred_test = np.empty((z_test.shape[0], N_boostraps, nlambdas))

    for i in range(N_boostraps):
        X_, z_ = resample(X_train, z_train)

        for j in range(nlambdas):
            lmb = lambdas[j]
            beta = np.linalg.pinv(X_.T @ X_+lmb*I) @ X_.T @ z_

            z_pred_train[:, i, j] = X_train@beta
            z_pred_test[:, i, j] = X_test@beta

            # # and then make the prediction
            # z_train_Ridge = X_train @ beta
            # z_test_Ridge = X_test @ beta

            # MSE_ridge[0, idx, i, j] = MSE(z_train,z_train_Ridge)
            # MSE_ridge[1, idx, i, j] = MSE(z_test,z_test_Ridge)

    z_train = np.reshape(z_train, (len(z_train), 1))
    z_test = np.reshape(z_test, (len(z_test), 1))

    for i in range(nlambdas):
        error[0, idx, i] = np.mean( np.mean((z_train - z_pred_train[:,:,i])**2, axis = 1, keepdims=True) )
        error[1, idx, i] = np.mean( np.mean((z_test - z_pred_test[:,:,i])**2, axis = 1, keepdims=True) )
        bias[0, idx, i] = np.mean( (z_train - np.mean(z_pred_train[:,:,i], axis = 1, keepdims=True))**2 )
        bias[1, idx, i] = np.mean( (z_test - np.mean(z_pred_test[:,:,i], axis = 1, keepdims=True))**2 )
        variance[0, idx, i] = np.mean( np.var(z_pred_train[:,:,i], axis = 1, keepdims=True) )
        variance[1, idx, i] = np.mean( np.var(z_pred_test[:,:,i], axis = 1, keepdims=True) )


for i in range(0, nlambdas, 2):
    fig, ax = plt.subplots()
    ax.set_title(f'Ridge regression with bootstrapping, lambda = {lambdas[i]:.3e}')
    ax.plot(poly_degrees, error[0, :, i], label="error training data")
    # ax.plot(poly_degrees, error[1, :, i], label="error test data")
    ax.plot(poly_degrees, bias[0, :, i], label="bias training data")
    # ax.plot(poly_degrees, bias[1, :, i], label="bias test data")
    ax.plot(poly_degrees, variance[0, :, i], label="variance training data")
    # ax.plot(poly_degrees, variance[1, :, i], label="variance test data")
    ax.set_xlabel('polynomial degree')
    ax.legend()
    plt.savefig('plots/exercise4_bootstrap_ridge_bvt.pdf')
    plt.show()

# MSE_ridge_mean = np.zeros((2, len(poly_degrees), nlambdas))
# for i, degree in enumerate(poly_degrees):
#     for j in range(nlambdas):
#         MSE_ridge_mean[0, i, j] = np.mean(MSE_ridge[0,i,:,j])
#         MSE_ridge_mean[1, i, j] = np.mean(MSE_ridge[1,i,:,j])


    # plt.figure()
    # plt.title(f'Ridge Regression {degree}')
    # plt.plot(np.log10(lambdas), MSE_ridge_mean[0][i][:], label = 'MSE Ridge train')
    # plt.plot(np.log10(lambdas), MSE_ridge_mean[1][i][:], label = 'MSE Ridge test')
    # plt.xlabel('log10(lambda)')
    # plt.ylabel('MSE')
    # plt.legend()
    # plt.show()



