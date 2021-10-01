import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


from common import FrankeFunction, MSE, create_X, scale, bias_variance, error_afo_polydeg, owncode_vs_sklear, error_afo_lambda

class Regression:
    def __init__(self, poly_degrees):
        self.poly_degrees = poly_degrees

    def simple_regression(self, x, y, z, lmbda):
        z = np.ravel(z)

        error = np.zeros((2, len(self.poly_degrees)))
        bias = np.zeros_like(error)
        variance = np.zeros_like(error)

        for idx, degree in enumerate(poly_degrees):
            # X_train, X_test, z_train, z_test = prepare_data_set(x, y, degree)

            X = create_X(x, y, n=degree)
            X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
            X_train, X_test = scale(X_train, X_test)

            z_pred_train = np.empty(z_train.shape[0])
            z_pred_test = np.empty(z_test.shape[0])

            self.fit(X_train, z_train, lmbda)
            z_pred_train = self.predict(X_train)
            z_pred_test = self.predict(X_test)

            error[0, idx] = np.mean((z_train - z_pred_train)**2)
            error[1, idx] = np.mean((z_test - z_pred_test)**2)
            bias[0, idx] = np.mean( (z_train - np.mean(z_pred_train))**2 )
            bias[1, idx] = np.mean( (z_test - np.mean(z_pred_test))**2 )
            variance[0, idx] = np.mean( np.var(z_pred_train) )
            variance[1, idx] = np.mean( np.var(z_pred_test) )
        return error, bias, variance

    def bootstrap(self, x, y, z, lambdas, N_bootstraps):
        z = np.ravel(z)

        n_lambdas = len(lambdas)

        error = np.zeros((2, len(self.poly_degrees), n_lambdas))
        bias = np.zeros_like(error)
        variance = np.zeros_like(error)

        for idx, degree in enumerate(poly_degrees):
            # X_train, X_test, z_train, z_test = prepare_data_set(x, y, degree)

            X = create_X(x, y, n=degree)
            X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
            X_train, X_test = scale(X_train, X_test)

            z_pred_train = np.empty((z_train.shape[0], N_boostraps, n_lambdas))
            z_pred_test = np.empty((z_test.shape[0], N_boostraps, n_lambdas))

            for i in range(N_boostraps):
                X_, z_ = resample(X_train, z_train)
                for j in range(n_lambdas):
                    self.fit(X_, z_, lambdas[j])
                    z_pred_train[:, i, j] = self.predict(X_train)
                    z_pred_test[:, i, j] = self.predict(X_test)

            z_train = np.reshape(z_train, (len(z_train), 1))
            z_test = np.reshape(z_test, (len(z_test), 1))

            for i in range(n_lambdas):
                error[0, idx, i] = MSE(z_train, z_pred_train[:,:,i])
                error[1, idx, i] = MSE(z_test, z_pred_test[:,:,i])
                # bias[0, idx, i] = bias(z_train, z_pred_train[:,:,i])
                # bias[1, idx, i] = bias(z_test, z_pred_test[:,:,i])
                # variance[0, idx, i] = variance(z_pred_train[:,:,i])
                # variance[1, idx, i] = variance(z_pred_test[:,:,i])

                bias[0, idx, i] = np.mean( (z_train - np.mean(z_pred_train[:,:,i], axis = 1, keepdims=True))**2 )
                bias[1, idx, i] = np.mean( (z_test - np.mean(z_pred_test[:,:,i], axis = 1, keepdims=True))**2 )
                variance[0, idx, i] = np.mean( np.var(z_pred_train[:,:,i], axis = 1, keepdims=True) )
                variance[1, idx, i] = np.mean( np.var(z_pred_test[:,:,i], axis = 1, keepdims=True) )
        return error, bias, variance


    def k_fold(self, xy, x, y, z, lambdas, k, method):
        kfold = KFold(n_splits = k, shuffle = True)

        error_folds = np.zeros((2, len(poly_degrees), k))
        error_sklearn = np.zeros(len(poly_degrees))
        error_folds_lmbd = np.zeros((2, len(lambdas), k))

        for idx, degree in enumerate(poly_degrees):
            # Decide degree on polynomial to fit
            poly = PolynomialFeatures(degree)
            for i, lmb in enumerate(lambdas):
                j = 0
                for train_inds, test_inds in kfold.split(xy):
                    xytrain = xy[train_inds]
                    z_train = z[train_inds]

                    xytest = xy[test_inds]
                    z_test = z[test_inds]

                    # Makes the design matrix for train and test data
                    X_train = poly.fit_transform(xytrain)
                    X_test = poly.fit_transform(xytest)

                    X_train, X_test = scale(X_train, X_test)

                    self.fit(X_train, z_train, 0)
                    self.fit(X_train, z_train, lambdas[i])
                    z_pred_train = self.predict(X_train)
                    z_pred_test = self.predict(X_test)

                    error_folds[0, idx, j] = MSE(z_train,z_pred_train)
                    error_folds[1, idx, j] = MSE(z_test,z_pred_test)

                    error_folds_lmbd[0, i, j] = MSE(z_train, z_pred_train)
                    error_folds_lmbd[1, i, j] = MSE(z_test, z_pred_test)
                    j += 1

                X = poly.fit_transform(xy)

                if method == 1:
                    method = LinearRegression(fit_intercept=False)

                elif method == 2:
                    method = Ridge(alpha=lmb, fit_intercept=False)

                elif method == 3:
                    method = Lasso(alpha=lmb, fit_intercept=False)

                error_sklearn_folds = cross_val_score(method, X, z, scoring='neg_mean_squared_error', cv=kfold)
                error_sklearn[idx] = np.mean(-error_sklearn_folds, axis = 0)

            error = np.mean(error_folds, axis = 2)
        error_lmbd = np.mean(error_folds_lmbd, axis = 2)
        return error, error_sklearn, error_lmbd

class OLSReg(Regression):
    def fit(self, X, z, *_):
        self.beta = np.linalg.pinv(X.T@X)@X.T@z

    def predict(self, X):
        return X@self.beta

class RidgeReg(Regression):
    def fit(self, X, z, lmbda):
        self.beta = np.linalg.pinv(X.T @ X + lmbda*np.eye(X.shape[1], X.shape[1])) @ X.T @ z

    def predict(self, X):
        return X@self.beta

class LassoReg(Regression):
    def fit(self, X, z, lmbda):
        self.regLasso = linear_model.Lasso(lmbda, fit_intercept=False)
        self.regLasso.fit(X, z)

    def predict(self, X):
        return self.regLasso.predict(X)

N = 100
poly_degrees = np.arange(1, 10)
N_boostraps = 100
k = 10
N_lambdas = 9
lambdas = np.logspace(-4, 4, N_lambdas)

np.random.seed(2018)

x = np.arange(0, 1, 1/N)
y = np.arange(0, 1, 1/N)

xy = np.zeros((N,2))
xy[:,0] = x
xy[:,1] = y

x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y)

# # OLS; SIMPLE, BOOTSTRAP, K-FOLD
# model = OLSReg(poly_degrees)
# error, bias, variance = model.simple_regression(x, y, z, 0)
# bias_variance(
#     poly_degrees, error, bias, variance,
#     title='OLS simple regression',
#     fname='plots/class-OLS-simple.pdf'
# )
# error, bias, variance = model.bootstrap(x, y, z, [0], N_boostraps)
# bias_variance(
#     poly_degrees, error, bias, variance,
#     title='OLS regression with bootstrapping',
#     fname='plots/class-OLS-bootstrap.pdf'
# )
# error, error_sklearn, *_ = model.k_fold(xy, x, y, z, [0], k, method = 1)
# error_afo_polydeg(
#     poly_degrees, error,
#     title='OLS regression with k-foldning',
#     fname='plots/class-OLS-k-fold.pdf',
# )
# owncode_vs_sklear(poly_degrees, error, error_sklearn,
#     title='Sklearn, OLS regression with k-foldning',
#     fname='plots/class-OLS-k-fold-sklearn.pdf')
#
# # RIDGE; SIMPLE, BOOTSTRAP, K-FOLD
# model = RidgeReg(poly_degrees)
# error, bias, variance = model.simple_regression(x, y, z, 0.5)
# bias_variance(
#     poly_degrees, error, bias, variance,
#     title='Ridge simple regression',
#     fname='plots/class-Ridge-simple.pdf'
# )
# error, bias, variance = model.bootstrap(x, y, z, lambdas, N_boostraps)
# for i in range(0, N_lambdas, 2):
#     bias_variance(
#         poly_degrees, error[:, :, i], bias[:, :, i], variance[:, :, i],
#         title=f'Ridge regression with bootstrapping, lambda = {lambdas[i]:.3e}',
#         fname=f'plots/class-Ridge-bootstrap_{i+1}.pdf'
#     )
# error, error_sklearn, error_lmbd = model.k_fold(xy, x, y, z, lambdas, k, method = 2)
# error_afo_polydeg(
#     poly_degrees, error,
#     title='Ridge regression with k-foldning',
#     fname='plots/class-Ridge-k-fold.pdf'
# )
# owncode_vs_sklear(poly_degrees, error, error_sklearn,
#     title='Sklearn, Ridge regression with k-foldning',
#     fname='plots/class-Ridge-k-fold-sklearn.pdf'
# )
#
# # plot_k_fold_lmbd(
# #     lambdas, error_lmbd,
# #     title='Ridge regression with k-foldning',
# #     fname='plots/class-Ridge-k-fold.pdf',
# # )
#
#
# # LASSO; SIMPLE, BOOTSTRAP, K-FOLD
# model = LassoReg(poly_degrees)
# error, bias, variance = model.simple_regression(x, y, z, 0.5)
# bias_variance(
#     poly_degrees, error, bias, variance,
#     title='Lasso simple regression',
#     fname='plots/class-Lasso-simple.pdf'
# )
#
# model = LassoReg(poly_degrees)
# error, bias, variance = model.bootstrap(x, y, z, lambdas, N_boostraps)
# for i in range(0, N_lambdas, 2):
#     bias_variance(
#         poly_degrees, error[:, :, i], bias[:, :, i], variance[:, :, i],
#         title=f'Lasso regression with bootstrapping, lambda = {lambdas[i]:.3e}',
#         fname=f'plots/class-Lasso-bootstrap_{i+1}.pdf'
#     )

# model = LassoReg(poly_degrees)
# error, bias, variance = model.bootstrap(x, y, z, lambdas, N_boostraps)
# for i in range(0, N_lambdas, 2):
#     error_afo_lambda(
#         poly_degrees, error[:, :, i], bias[:, :, i], variance[:, :, i],
#         title=f'Lasso regression with bootstrapping, lambda = {lambdas[i]:.3e}',
#         fname=f'plots/class-Lasso-bootstrap_{i+1}.pdf'
#     )
model = LassoReg(poly_degrees)
# error, error_sklearn, error_lmbd = model.k_fold(xy, x, y, z, lambdas, k, method = 3)
# for i in range(0, N_lambdas, 2):
#     error_afo_polydeg(
#         poly_degrees, error,
#         title='Lasso regression with k-foldning',
#         fname='plots/class-Lasso-k-fold_lmb.pdf',
#     )
*_, error_lmbd = model.k_fold(xy, x, y, z, lambdas, k, method = 3)
for i in range(0, N_lambdas, 2):
    error_afo_lambda(
        lambdas, error_lmbd,
        title='Lasso regression with k-foldning as function of lambda',
        fname='plots/class-Lasso-k-fold_lmb.pdf',
    )


