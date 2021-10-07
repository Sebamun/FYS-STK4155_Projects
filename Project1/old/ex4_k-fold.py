from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
import sklearn.linear_model as skl

from common import FrankeFunction, create_X, MSE, scale

# For nice plots
plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

np.random.seed(2020)

N = 100
x = np.arange(0, 1, 1/N)
y = np.arange(0, 1, 1/N)

xy = np.zeros((N,2))
xy[:,0] = x
xy[:,1] = y

x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y)

# Decide degree on polynomial to fit
poly = PolynomialFeatures(degree = 4)

nlambdas = 20
lambdas = np.logspace(-4, 4, nlambdas)

# Initialize a KFold instance, number of splitted parts
k = 10 # try 5 - 10 folders
#Provides train/test indices to split data in train/test sets.
#Split dataset into k consecutive folds (without shuffling by default).
kfold = KFold(n_splits = k)

# Perform the cross-validation to estimate MSE
scores_KFold = np.zeros((2, nlambdas, k))

for i in range(nlambdas):
    j = 0
    for train_inds, test_inds in kfold.split(xy):
        xytrain = xy[train_inds]
        z_train = z[train_inds]

        xytest = xy[test_inds]
        z_test = z[test_inds]

        # Makes the design matrix for train data
        X_train = poly.fit_transform(xytrain)

        # Makes the design matrix for test data
        X_test = poly.fit_transform(xytest)

        X_train, X_test = scale(X_train, X_test)

        I = np.eye(X_train.shape[1], X_test.shape[1]) #Return a 2-D array with ones on the diagonal and zeros elsewhere
        lmb = lambdas[i]
        Ridgebeta = np.linalg.pinv(X_train.T @ X_train+lmb*I) @ X_train.T @ z_train #pinv/ inv?

        # and then make the prediction
        z_train_Ridge = X_train @ Ridgebeta
        z_test_Ridge = X_test @ Ridgebeta

        scores_KFold[0, i, j] = MSE(z_train,z_train_Ridge)
        scores_KFold[1, i, j] = MSE(z_test,z_test_Ridge)

        j += 1

estimated_mse_KFold = np.mean(scores_KFold, axis = 2)


plt.figure()
plt.title('Ridge regression with k-folding')
plt.plot(np.log10(lambdas), estimated_mse_KFold[0], label = 'MSE, train data')
plt.plot(np.log10(lambdas), estimated_mse_KFold[1], label = 'MSE, test data')
plt.xlabel('log10(lambda)')
plt.ylabel('mse')
plt.legend()
plt.savefig('plots/exercise4_k-fold_ridge_MSEvsLmb.pdf')
plt.show()