import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import numpy as np
from common import FrankeFunction, create_X, MSE

# For nice plots
plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

np.random.seed(2018)

N = 100
N_boostraps = 50
poly_degrees = np.arange(1, 8)
nlambdas = 10

# Making meshgrid of datapoints and compute Franke's function
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y).ravel()

lambdas = np.logspace(-3, 1, nlambdas)

error = np.zeros((2, len(poly_degrees), nlambdas))
bias = np.zeros_like(error)
variance = np.zeros_like(error)

for idx, degree in enumerate(poly_degrees):
    X = create_X(x, y, n=degree)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.2)

    # scale all columns except the first (which is all ones)
    scaler = StandardScaler()
    scaler.fit(X_train[:, 1:])
    X_train[:, 1:] = scaler.transform(X_train[:, 1:])
    X_test[:, 1:] = scaler.transform(X_test[:, 1:])

    # The following (m x n_bootstraps) matrix holds the column vectors z_pred
    # for each bootstrap iteration.
    zpredictLasso_train = np.empty((z_train.shape[0], N_boostraps, nlambdas))
    zpredictLasso_test = np.empty((z_test.shape[0], N_boostraps, nlambdas))

    for i in range(N_boostraps):
        X_, z_ = resample(X_train, z_train)
        for j in range(nlambdas):
            RegLasso = linear_model.Lasso(lambdas[j], fit_intercept=False) #sjekk andre parametere
            RegLasso.fit(X_, z_)
            zpredictLasso_train[:,i,j] = RegLasso.predict(X_train)
            zpredictLasso_test[:,i,j] = RegLasso.predict(X_test)

    z_train = np.reshape(z_train, (len(z_train), 1))
    z_test = np.reshape(z_test, (len(z_test), 1))

    for i in range(nlambdas):
        error[0, idx, i] = np.mean((z_train - zpredictLasso_train[:,:,i])**2)
        # endre resten til enklere form
        error[1, idx, i] = np.mean( np.mean((z_test - zpredictLasso_test[:,:,i])**2, axis = 1, keepdims=True) )
        bias[0, idx, i] = np.mean( (z_train - np.mean(zpredictLasso_train[:,:,i], axis = 1, keepdims=True))**2 )
        bias[1, idx, i] = np.mean( (z_test - np.mean(zpredictLasso_test[:,:,i], axis = 1, keepdims=True))**2 )
        variance[0, idx, i] = np.mean( np.var(zpredictLasso_train[:,:,i], axis = 1, keepdims=True) )
        variance[1, idx, i] = np.mean( np.var(zpredictLasso_test[:,:,i], axis = 1, keepdims=True) )


for i in range(0, nlambdas, 2):
    fig, ax = plt.subplots()
    ax.set_title(f'Lasso regression with bootstrapping, lambda = {lambdas[i]:.3e}')
    ax.plot(poly_degrees, error[0, :, i], label="error training data")
    # ax.plot(poly_degrees, error[1, :, i], label="error test data")
    ax.plot(poly_degrees, bias[0, :, i], label="bias training data")
    # ax.plot(poly_degrees, bias[1, :, i], label="bias test data")
    ax.plot(poly_degrees, variance[0, :, i], label="variance training data")
    # ax.plot(poly_degrees, variance[1, :, i], label="variance test data")
    ax.set_xlabel('polynomial degree')
    ax.legend()
    plt.savefig('plots/exercise5_bootstrap_lasso_bvt.pdf')
    plt.show()