import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

def FrankeFunction(x, y):
    assert len(x.shape) == 2
    assert x.shape[0] == x.shape[1]
    assert x.shape == y.shape
    N = x.shape[0]
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + np.random.normal(0, 0.5, (N,N))

# def prepare_data_set(x, y, z, degree):
#     X = create_X(x, y, n=degree)
#     X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
#     X_train, X_test = scale(X_train, X_test)

def create_X(x, y, n):
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

def MSE(y_data, y_model):
    return np.mean((y_data - y_model)**2)

def bias(y_data, y_model):
    return np.mean( (y_data - np.mean(y_model, axis = 1, keepdims=True))**2 )

def variance(y):
    return np.mean( np.var(y, axis = 1, keepdims=True) )

def scale(X_train, X_test):
    """Scale all columns except the first (which is all ones)"""
    scaler = StandardScaler()
    scaler.fit(X_train[:, 1:])
    X_train[:, 1:] = scaler.transform(X_train[:, 1:])
    X_test[:, 1:] = scaler.transform(X_test[:, 1:])
    return X_train, X_test

def bias_variance(poly_degrees, MSE, bias, variance, title, fname):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.plot(poly_degrees, MSE[0], label='MSE training data')
    ax.plot(poly_degrees, bias[0], label='bias training data')
    ax.plot(poly_degrees, variance[0], label='variance training data')
    ax.set_xlabel('Polynomial degree')
    ax.legend()
    plt.savefig(fname)
    plt.close(fig)

def error_afo_polydeg(poly_degrees, MSE, title, fname):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.plot(poly_degrees, MSE[0], label = 'MSE, train data')
    ax.plot(poly_degrees, MSE[1], label = 'MSE, test data')
    ax.set_xlabel('polynomial degree')
    ax.set_ylabel('mse')
    ax.legend()
    plt.savefig(fname)

def owncode_vs_sklear(poly_degrees, MSE, MSE_sklearn, title, fname):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.plot(poly_degrees, MSE_sklearn, label = 'MSE, sklearn')
    ax.plot(poly_degrees, MSE[1], 'r--', label = 'MSE test, own code')
    ax.set_xlabel('polynomial degree')
    ax.set_ylabel('mse')
    ax.legend()
    plt.savefig(fname)

def error_afo_lambda(lambdas, error_lmbd, title, fname):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.plot(np.log10(lambdas), error_lmbd[0], label = 'MSE, train data')
    ax.plot(np.log10(lambdas), error_lmbd[1], label = 'MSE, test data')
    ax.set_xlabel('log10(lambda)')
    ax.set_ylabel('mse')
    ax.legend()
    plt.savefig(fname)
