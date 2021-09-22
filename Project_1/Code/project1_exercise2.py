import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
from numpy.random import randint, randn
from time import time
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

# For nice plots
plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

np.random.seed(2018)

N = 25
N_boostraps = 100
poly_degrees = np.arange(1, 10)
noise = 0.1

# Make data set.
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + np.random.normal(-0.1, 0.1, N)

def create_X(x, y, n):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N, l))

	for i in range(1, n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X

x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
# x = np.arange(0, 1, 0.05)
# y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y).ravel()

MSE = np.zeros((2, len(poly_degrees), N_boostraps))
error = np.zeros((2, len(poly_degrees)))
bias = np.zeros((2, len(poly_degrees)))
variance = np.zeros((2, len(poly_degrees)))

for idx, degree in enumerate(poly_degrees):
    X = create_X(x, y, degree)

    # Hold out some test data that is never used in training.
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    # The following (m x n_bootstraps) matrix holds the column vectors z_pred
    # for each bootstrap iteration.
    z_pred_train = np.empty((z_train.shape[0], N_boostraps))
    z_pred_test = np.empty((z_test.shape[0], N_boostraps))
    for i in range(N_boostraps):
        X_, z_ = resample(X_train, z_train) # Collects value for each bootstrap iteration.

        # Ordinary least squares
        beta = np.linalg.pinv(X_.T@X_)@X_.T@z_  # train/fit the model

        # Evaluate the new model on the same test data each time.
        z_pred_train[:, i] = X_train@beta
        z_pred_test[:, i] = X_test@beta

        MSE[0, idx, i] = mean_squared_error(z_train, z_pred_train[:, i])
        MSE[1, idx, i] = mean_squared_error(z_test, z_pred_test[:, i])

    z_train = np.reshape(z_train, (len(z_train), 1))
    z_test = np.reshape(z_test, (len(z_test), 1))
    error[0, idx] = np.mean( np.mean((z_train - z_pred_train)**2, axis = 1, keepdims=True) )
    error[1, idx] = np.mean( np.mean((z_test - z_pred_test)**2, axis = 1, keepdims=True) )
    bias[0, idx] = np.mean( (z_train - np.mean(z_pred_train, axis = 1, keepdims=True))**2 )
    bias[1, idx] = np.mean( (z_test - np.mean(z_pred_test, axis = 1, keepdims=True))**2 )
    variance[0, idx] = np.mean( np.var(z_pred_train, axis = 1, keepdims=True) )
    variance[1, idx] = np.mean( np.var(z_pred_test, axis = 1, keepdims=True) )

MSE_mean = np.zeros((2, len(poly_degrees)))
for idx, degree in enumerate(poly_degrees):
    MSE_mean[0, idx] = np.mean(MSE[0, idx])
    MSE_mean[1, idx] = np.mean(MSE[1, idx])

fig, ax = plt.subplots()
ax.set_title('Mean Square Error')
ax.plot(poly_degrees, MSE_mean[0], label="Training data")
ax.plot(poly_degrees, MSE_mean[1], label="Test data")
ax.set_xlabel('polynomial degree')
ax.set_ylabel('MSE')
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.set_title('bias-variance tradeoff')
ax.plot(poly_degrees, error[1], label="error training data")
ax.plot(poly_degrees, bias[1], label="bias training data")
ax.plot(poly_degrees, variance[1], label="variance training data")
ax.set_xlabel('polynomial degree')
ax.legend()
plt.show()
