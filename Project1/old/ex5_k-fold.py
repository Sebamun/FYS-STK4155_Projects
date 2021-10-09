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
from sklearn import linear_model

from common import FrankeFunction, create_X, MSE, scale

# For nice plots
plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

np.random.seed(2018)

N = 100
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))

xy = np.zeros((N,2))
xy[:,0] = x
xy[:,1] = y

x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y)

# Decide degree on polynomial to fit
poly = PolynomialFeatures(degree = 10)

nlambdas = 20
lambdas = np.logspace(-2, 1, nlambdas)

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

        RegLasso = linear_model.Lasso(lambdas[i], fit_intercept=False)
        RegLasso.fit(X_train, z_train)

        # and then make the prediction
        zpredictLasso_train = RegLasso.predict(X_train)
        zpredictLasso_test = RegLasso.predict(X_test)

        scores_KFold[0, i, j] = MSE(z_train, zpredictLasso_train)
        scores_KFold[1, i, j] = MSE(z_test, zpredictLasso_test)

        j += 1

estimated_mse_KFold = np.mean(scores_KFold, axis = 2)


plt.figure()
plt.title('Lasso regression with k-folding')
plt.plot(np.log10(lambdas), estimated_mse_KFold[0], label = 'MSE, train data')
plt.plot(np.log10(lambdas), estimated_mse_KFold[1], label = 'MSE, test data')
plt.xlabel('log10(lambda)')
plt.ylabel('mse')
plt.legend()
plt.savefig('plots/exercise5_k-fold_lasso_MSEvsLmb.pdf')
plt.show()