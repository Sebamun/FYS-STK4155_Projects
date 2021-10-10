import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def FrankeFunction(x, y):
    assert len(x.shape) == 2
    assert x.shape[0] == x.shape[1]
    assert x.shape == y.shape
    N = x.shape[0]
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + np.random.normal(0, 0.1, (N,N))

def prepare_data_set(x, y, z, degree, scale_data):
    z = np.ravel(z)
    X = create_X(x, y, n=degree)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2) #, random_state=2018)
    if scale_data:
        X_train, X_test = scale(X_train, X_test)
    return X_train, X_test, z_train, z_test

def create_X(x, y, n):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2) # Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)

    return X

def scale(X_train, X_test=None):
    """Scale all columns except the first (which is all ones)"""
    scaler = StandardScaler()
    scaler.fit(X_train[:, 1:])
    X_train[:, 1:] = scaler.transform(X_train[:, 1:])
    if X_test is not None:
        X_test[:, 1:] = scaler.transform(X_test[:, 1:])
    return X_train, X_test


def MSE(y_data, y_model):
    return np.mean((y_data - y_model)**2)

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

def bias_(y_data, y_model):
    if y_model.ndim == 1:
        return np.mean((y_data - np.mean(y_model))**2)
    else:
        return np.mean((y_data - np.mean(y_model, axis = 1, keepdims=True))**2)

def variance_(y):
    if y.ndim == 1:
        return np.mean(np.var(y))
    else:
        return np.mean(np.var(y, axis = 1, keepdims=True))

def compute_means_over_lambda(bias, variance, N_lambdas):
    mean_bias = np.zeros_like(bias[:, :, 0])
    mean_variance = np.zeros_like(variance[:, :, 0])
    for i in range(0, N_lambdas):
        mean_bias += bias[:, :, i]
        mean_variance += variance[:, :, i]
    mean_bias = mean_bias/N_lambdas
    mean_variance = mean_variance/N_lambdas
    return mean_bias, mean_variance
