import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from common import (MSE, R2, bias_, variance_, prepare_data_set,
scale)

class Regression:
    def __init__(self, poly_degrees):
        self.poly_degrees = poly_degrees

    def simple_regression(self, x, y, z, lmbda, scale=False):
        X_array = np.zeros(len(self.poly_degrees), dtype = object)
        beta_array = np.zeros(len(self.poly_degrees), dtype = object)
        error = np.zeros((2, len(self.poly_degrees)))
        r2 = np.zeros_like(error)
        bias = np.zeros_like(error)
        variance = np.zeros_like(error)

        for idx, degree in enumerate(self.poly_degrees):
            X, X_train, X_test, z_train, z_test = prepare_data_set(
                x, y, z, degree, scale)

            X_array[idx] = X

            z_pred_train = np.empty(z_train.shape[0])
            z_pred_test = np.empty(z_test.shape[0])

            self.fit(X_train, z_train, lmbda)
            beta_array[idx] = self.beta

            z_pred_train = self.predict(X_train)
            z_pred_test = self.predict(X_test)

            error[0, idx] = MSE(z_train, z_pred_train)
            error[1, idx] = MSE(z_test, z_pred_test)
            r2[0, idx] = R2(z_train, z_pred_train)
            r2[1, idx] = R2(z_test, z_pred_test)
            bias[0, idx] = bias_(z_train, z_pred_train)
            bias[1, idx] = bias_(z_test, z_pred_test)
            variance[0, idx] = variance_(z_pred_train)
            variance[1, idx] = variance_(z_pred_test)

        return error, r2, bias, variance, X_array, beta_array

    def bootstrap(self, x, y, z, lambdas, N_bootstraps):
        n_lambdas = len(lambdas)

        error = np.zeros((2, len(self.poly_degrees), n_lambdas))
        bias = np.zeros_like(error)
        variance = np.zeros_like(error)

        for idx, degree in enumerate(self.poly_degrees):
            X, X_train, X_test, z_train, z_test = prepare_data_set(
                x, y, z, degree, scale_data=True)

            z_pred_train = np.empty((z_train.shape[0], N_bootstraps, n_lambdas))
            z_pred_test = np.empty((z_test.shape[0], N_bootstraps, n_lambdas))

            for i in range(N_bootstraps):
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
                bias[0, idx, i] = bias_(z_train, z_pred_train[:,:,i])
                bias[1, idx, i] = bias_(z_test, z_pred_test[:,:,i])
                variance[0, idx, i] = variance_(z_pred_train[:,:,i])
                variance[1, idx, i] = variance_(z_pred_test[:,:,i])
        return error, bias, variance

    def k_fold(self, xy, x, y, z, lambdas, k):
        kfold = KFold(n_splits = k)

        error_folds = np.zeros((2, len(self.poly_degrees), len(lambdas), k))
        error_sklearn = np.zeros((len(self.poly_degrees), len(lambdas)))

        for idx, degree in enumerate(self.poly_degrees):
            # Decide degree on polynomial to fit
            poly = PolynomialFeatures(degree)
            X = poly.fit_transform(xy)
            X, _ = scale(X)

            for i, lmb in enumerate(lambdas):
                j = 0
                for train_inds, test_inds in kfold.split(xy, y=z):
                    X_train = X[train_inds]
                    X_test = X[test_inds]
                    z_train = z[train_inds]
                    z_test = z[test_inds]

                    self.fit(X_train, z_train, lambdas[i])
                    z_pred_train = self.predict(X_train)
                    z_pred_test = self.predict(X_test)

                    error_folds[0, idx, i, j] = MSE(z_train, z_pred_train)
                    error_folds[1, idx, i, j] = MSE(z_test, z_pred_test)
                    j += 1

                method = self.sklearn_model(lmb)

                error_sklearn_folds = cross_val_score(
                    method, X, z, scoring='neg_mean_squared_error', cv=kfold
                )
                error_sklearn[idx, i] = np.mean(-error_sklearn_folds, axis=0)

        error_folds = np.mean(error_folds, axis=3)
        return error_folds, error_sklearn

class OLSReg(Regression):
    def fit(self, X, z, *_):
        self.beta = np.linalg.pinv(X.T@X)@X.T@z

    def predict(self, X):
        return X@self.beta

    def sklearn_model(self, _):
        return LinearRegression(fit_intercept=False)

class RidgeReg(Regression):
    def fit(self, X, z, lmbda):
        self.beta = np.linalg.pinv(X.T @ X + lmbda*np.eye(X.shape[1], X.shape[1])) @ X.T @ z

    def predict(self, X):
        return X@self.beta

    def sklearn_model(self, lmb):
        return Ridge(alpha=lmb, fit_intercept=False)

class LassoReg(Regression):
    def fit(self, X, z, lmbda):
        self.regLasso = Lasso(
            lmbda, fit_intercept=False, max_iter=1e6, tol=1e-2
        )
        self.regLasso.fit(X, z)

    def predict(self, X):
        return self.regLasso.predict(X)

    def sklearn_model(self, lmb):
        return Lasso(alpha=lmb, fit_intercept=False, max_iter=1e6, tol=1e-2)

