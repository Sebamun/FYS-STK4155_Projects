from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import sklearn.linear_model as skl

# For nice plots
plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

np.random.seed(2018)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + np.random.normal(-0.1, 0.1, (N,N))

def create_X(x, y, n ):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

# Making meshgrid of datapoints and compute Franke's function
N = 200
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y)

z = np.ravel(z)

poly_degrees = np.arange(2, 7)

nlambdas = 20
lambdas = np.logspace(-4, 1, nlambdas)
MSE_ridge = np.zeros((2, len(poly_degrees), nlambdas))
MSE_OLS = np.zeros((2, len(poly_degrees)))

for idx, degree in enumerate(poly_degrees):
# Repeat now for Ridge regression and various values of the regularization parameter

    X = create_X(x, y, n = degree)

    # split in training and test data
    split_size = 0.2
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.2)

    #Scaling:
    X_train_new = np.delete(X_train, 0, 1) # We remove the first column (consisting of ones).
    X_test_new = np.delete(X_test, 0, 1)

    scaler = StandardScaler() # Scaling to unit variance.
    scaler.fit(X_train_new)
    scaler.fit(X_test_new)

    X_train_new = scaler.transform(X_train_new) # Subtracts mean and divide over standard deviation
    X_test_new = scaler.transform(X_test_new) # and make our data unitless.

    array_ones_train = np.ones((int(N*(1-split_size)) * N, 1)) # 80%
    array_ones_test = np.ones((int(N*split_size) * N, 1)) # 20%
    X_train = np.append(array_ones_train, X_train_new, axis = 1) # Here we create our new matrcies,
    X_test = np.append(array_ones_test, X_test_new, axis = 1) # Where the ones are appended in the first column.


    # Ordinary least squares, NO TRAINING ON TEST DATA
    beta = np.linalg.pinv(X_train.T@X_train)@X_train.T@z_train  # train/fit the model
    z_predict_train = X_train@beta
    z_predict_test = X_test@beta

    MSE_OLS[0, idx] = MSE(z_train, z_predict_train)
    MSE_OLS[1, idx] = MSE(z_test, z_predict_test)

    # MSE_predict_train = np.zeros(nlambdas)
    # MSE_predict_test = np.zeros(nlambdas)
    I = np.eye(X_train.shape[1], X_test.shape[1]) #Return a 2-D array with ones on the diagonal and zeros elsewhere
    for i in range(nlambdas):
        lmb = lambdas[i]
        Ridgebeta = np.linalg.inv(X_train.T @ X_train+lmb*I) @ X_train.T @ z_train

        # and then make the prediction
        z_train_Ridge = X_train @ Ridgebeta
        z_test_Ridge = X_test @ Ridgebeta
        # MSE_predict_train[i, degree] = MSE(z_train,z_train_Ridge)
        # MSE_predict_test[i, degree] = MSE(z_test,z_test_Ridge)
        MSE_ridge[0, idx, i] = MSE(z_train,z_train_Ridge)
        MSE_ridge[1, idx, i] = MSE(z_test,z_test_Ridge)


# Now plot the results
for idx, degree in enumerate(poly_degrees):
    plt.figure()
    plt.title(f'Ridge Regression for polynomial degree {degree}')
    plt.plot(np.log10(lambdas), MSE_ridge[1, idx, :], label = 'MSE Ridge train')
    plt.plot(np.log10(lambdas), MSE_ridge[0, idx, :], 'r--', label = 'MSE Ridge Test')
    plt.xlabel('log10(lambda)')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig(f'exersice4_Ridge_MSEvsLambda_poly{degree}.pdf')
    plt.show()

fig, ax = plt.subplots()
ax.set_title('Mean Square Error, training data')
ax.plot(poly_degrees, MSE_OLS[0], label="Training data OLS")
for i in range(0, nlambdas, 5):
    ax.plot(poly_degrees, MSE_ridge[0, :, i], label=f"Training data Ridge, lambda = {lambdas[i]:.3e}")
ax.set_xlabel('polynomial degree')
ax.set_ylabel('MSE')
ax.legend()
plt.savefig('exersice4_Ridge_MSEvsPoly.pdf')
plt.show()