import numpy as np
from regression_methods import OLSReg, RidgeReg, LassoReg
from common import FrankeFunction
from plot import (MSE_R2, bias_variance, error_of_polydeg, owncode_vs_sklearn,
error_of_lambda)

N = 50
poly_degrees = np.arange(1, 10)
N_boostraps = 100
k = 5
N_lambdas = 9
lambdas = np.logspace(-4, 1, N_lambdas)

np.random.seed(2018)

x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)

xy = np.zeros((N,2))
xy[:,0] = x
xy[:,1] = y

x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y)

""" OLS; SIMPLE, BOOTSTRAP, K-FOLD """
def produce_results_OLS():
    model = OLSReg(poly_degrees)
    error, r2, bias, variance = model.simple_regression(x, y, z, 0, False)
    MSE_R2(
        poly_degrees, error, r2,
        title='OLS regression with simple resampling',
        fname=f'plots/exercise1_MSE_R2score{N}.pdf'
    )
    bias_variance(
        poly_degrees, error, bias, variance,
        title='OLS regression with simple resampling',
        fname=f'plots/exercise2_bias_variance{N}.pdf'
    )

    error, bias, variance = model.bootstrap(x, y, z, [0], N_boostraps)
    bias_variance(
        poly_degrees, error, bias, variance,
        title='OLS regression with bootstrapping',
        fname=f'plots/exercise2_bootstrap{N}.pdf'
    )

    error, error_sklearn = model.k_fold(xy, x, y, z, [0], k)
    error_of_polydeg(
        poly_degrees, error,
        title='OLS regression with k-foldning',
        fname=f'plots/exercise3_k-fold{N}.pdf'
    )

    owncode_vs_sklearn(poly_degrees, error[:, :, 0], error_sklearn[:, 0],
        title='OLS regression with k-foldning',
        fname=f'plots/exercise3_k-fold_sklearn{N}.pdf'
    )

""" RIDGE; SIMPLE, BOOTSTRAP, K-FOLD """
def produce_results_ridge():
    model = RidgeReg(poly_degrees)
    lmb = 0.5
    error, _, bias, variance = model.simple_regression(x, y, z, lmb, True)
    bias_variance(
        poly_degrees, error, bias, variance,
        title=r'Ridge regression with simple resampling for $\lambda = $' + f'{lmb}',
        fname=f'plots/exercise4_bias_variance_lmb_{lmb}_{N}.pdf'
    )
    error, bias, variance = model.bootstrap(x, y, z, lambdas, N_boostraps)
    for i in range(0, N_lambdas, 2):
        bias_variance(
            poly_degrees, error[:, :, i], bias[:, :, i], variance[:, :, i],
            title=r'Ridge regression with bootstrapping for $\lambda = $' + f'{lambdas[i]:.2e}',
            fname=f'plots/exercise4-bootstrap_{i+1}_{N}.pdf'
        )
    error, error_sklearn = model.k_fold(xy, x, y, z, lambdas, k)
    deg_idx = 4
    lmb_idx = 3
    error_of_polydeg(
        poly_degrees, error[:, :, lmb_idx],
        title=r'Ridge regression with k-foldning for $\lambda = $' + f'{lambdas[lmb_idx]:.2e}',
        fname=f'plots/exercise4_k-fold_lmb_{lambdas[lmb_idx]:.2e}_{N}.pdf'
    )
    owncode_vs_sklearn(poly_degrees, error[:, :, lmb_idx], error_sklearn[:, lmb_idx],
        title='Ridge regression with k-foldning for $\lambda = $' + f'{lambdas[lmb_idx]:.2e}',
        fname=f'plots/exercise4_k-fold_sklearn_lmb_{lambdas[lmb_idx]:.2e}_{N}.pdf')

    error_of_lambda(
        lambdas, error[:, deg_idx],
        title=f'Ridge regression with k-foldning for polynomial degree {poly_degrees[deg_idx]:.2e}',
        fname=f'plots/exercise4_k-fold_lambda_deg_{poly_degrees[deg_idx]:.2e}_{N}.pdf'
    )

""" LASSO; SIMPLE, BOOTSTRAP, K-FOLD """
def produce_results_lasso():
    lmb = 0.5
    model = LassoReg(poly_degrees)
    error, _, bias, variance = model.simple_regression(x, y, z, lmb, True)
    bias_variance(
        poly_degrees, error, bias, variance,
        title=r'Lasso regression with simple resampling for $\lambda = $' + f'{lmb}',
        fname=f'plots/exercise5_bias_variance_lmb_{lmb}_{N}.pdf'
    )

    error, bias, variance = model.bootstrap(x, y, z, lambdas, N_boostraps)
    for i in range(0, N_lambdas, 2):
        bias_variance(
            poly_degrees, error[:, :, i], bias[:, :, i], variance[:, :, i],
            title=r'Lasso regression with bootstrapping for $\lambda = $' + f'{lambdas[i]:.2e}',
            fname=f'plots/exercise5-bootstrap_{i+1}_{N}.pdf'
        )
    error, error_sklearn = model.k_fold(xy, x, y, z, lambdas, k)
    deg_idx = 4
    lmb_idx = 3
    error_of_polydeg(
        poly_degrees, error[:, :, lmb_idx],
        title=r'Lasso regression with k-foldning for $\lambda = $' + f'{lambdas[lmb_idx]:.2e}',
        fname=f'plots/exercise5_k-fold_lmb_{lambdas[lmb_idx]:.2e}_{N}.pdf'
    )

    owncode_vs_sklearn(poly_degrees, error[:, :, lmb_idx], error_sklearn[:, lmb_idx],
        title='Lasso regression with k-foldning for $\lambda = $' + f'{lambdas[lmb_idx]:.2e}',
        fname=f'plots/exercise5_k-fold_sklearn_lmb_{lambdas[lmb_idx]:.2e}_{N}.pdf'
    )

    error_of_lambda(
        lambdas, error[:, deg_idx],
        title=f'Lasso regression with k-foldning for polynomial degree {poly_degrees[deg_idx]:.2e}',
        fname=f'plots/exercise5_k-fold_lambda_deg_{poly_degrees[deg_idx]:.2e}_{N}.pdf'
    )

produce_results_OLS()
produce_results_ridge()
produce_results_lasso()