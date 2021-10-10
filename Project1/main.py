import numpy as np
from regression_methods import OLSReg, RidgeReg, LassoReg
from common import FrankeFunction
from plot import (MSE_R2, bias_variance, error_of_polydeg, owncode_vs_sklearn, test_vs_train,
error_of_lambda, variance_of_lambda, bias_variance_of_lambda, bias_variance_poly)

N = 20
poly_degrees = np.arange(1, 15)
N_boostraps = 20
k = 5
N_lambdas = 20
lambdas = np.logspace(-4, 2, N_lambdas)

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
    error, r2, bias, variance = model.simple_regression(x, y, z, 0, True)
    MSE_R2(
        poly_degrees, error, r2, N,
        title=f'OLS, MSE vs R2, {N}x{N}',
        fname=f'plots/OLS/OLS_MSE_R2score{N}.pdf'
    )
    test_vs_train(
        poly_degrees, error, N,
        title=f'OLS regression, no resampling, {N}x{N} datapoints',
        fname=f'plots/OLS/OLS_Test_vs_train_{N}.pdf'
    )

    error, bias, variance = model.bootstrap(x, y, z, [0], N_boostraps)
    bias_variance(
        poly_degrees, error, bias, variance, N,
        title='OLS regression with bootstrapping',
        fname=f'plots/OLS/OLS_BiasVariance_BS_{N}.pdf'
    )

    error, error_sklearn = model.k_fold(xy, x, y, z, [0], k)
    error_of_polydeg(
        poly_degrees, error,
        title='OLS regression with k-foldning',
        fname=f'plots/OLS/OLS_kfold_MSE{N}.pdf'
    )

    owncode_vs_sklearn(poly_degrees, error[:, :, 0], error_sklearn[:, 0],
        title='OLS, k-foldning. Own code VS sklearn',
        fname=f'plots/OLS/OLS_k-fold_sklearn{N}.pdf'
    )

""" RIDGE; SIMPLE, BOOTSTRAP, K-FOLD """
def produce_results_ridge():

    model = RidgeReg(poly_degrees)
    '''
    lmb = 0.5
    error, _, bias, variance = model.simple_regression(x, y, z, lmb, True)
    bias_variance(
        poly_degrees, error, bias, variance,
        title=r'Ridge regression with simple resampling for $\lambda = $' + f'{lmb}',
        fname=f'plots/exercise4_bias_variance_lmb_{lmb}_{N}.pdf'
        )
    '''

    error, bias, variance = model.bootstrap(x, y, z, lambdas, N_boostraps)
    mean_bias = np.zeros_like(bias[:, :, 0])
    mean_variance = np.zeros_like(variance[:, :, 0])
    for i in range(0, N_lambdas):
        mean_bias += bias[:, :, i]
        mean_variance += variance[:, :, i]
    mean_bias = mean_bias/N_lambdas
    mean_variance = mean_variance/N_lambdas
    bias_variance_poly(
        poly_degrees, mean_bias, bias[:, :, i], mean_variance,
        title=r'Bias Variance trend, taken as mean over all $\lambda$, )' + f'{N}x{N}',
        fname=f'plots/Ridge/Ridge_BiasVariance_BS_mean_{N}.pdf'
    )
    deg_idx = 1
    lmb_idx = 1
    bias_variance_of_lambda(
        lambdas, variance[:, deg_idx], bias[:, deg_idx],
        title=f'Ridge bias-variance, {N}x{N}, p={poly_degrees[deg_idx]}',
        fname=f'plots/Ridge/Ridge_BiasVariance_BS_lambda_{poly_degrees[deg_idx]:.2e}_{N}.pdf'
    )

    error, error_sklearn = model.k_fold(xy, x, y, z, lambdas, k)
    variance_of_lambda(
        lambdas, variance[:, deg_idx],
        title=f'Ridge regression with k-foldning for polynomial degree {poly_degrees[deg_idx]}',
        fname=f'plots/Ridge/kfold/Ridge_variance_k-fold_lambda_deg_{poly_degrees[deg_idx]:.2e}_{N}.pdf'
    )
    error_of_polydeg(
        poly_degrees, error[:, :, lmb_idx],
        title=r'Ridge regression with k-foldning for $\lambda = $' + f'{lambdas[lmb_idx]:.2e}',
        fname=f'plots/Ridge/kfold/Ridge_k-fold_lmb_{lambdas[lmb_idx]:.2e}_{N}.pdf'
    )
    owncode_vs_sklearn(poly_degrees, error[:, :, lmb_idx], error_sklearn[:, lmb_idx],
        title='Ridge regression with k-foldning for $\lambda = $' + f'{lambdas[lmb_idx]:.2e}',
        fname=f'plots/Ridge/kfold/Ridge_k-fold_sklearn_lmb_{lambdas[lmb_idx]:.2e}_{N}.pdf')

    error_of_lambda(
        lambdas, error[:, deg_idx],
        title=f'Ridge regression with k-foldning for polynomial degree {poly_degrees[deg_idx]}',
        fname=f'plots/Ridge/kfold/Ridge_k-fold_lambda_deg_{poly_degrees[deg_idx]:.2e}_{N}.pdf'
    )

""" LASSO; SIMPLE, BOOTSTRAP, K-FOLD """
def produce_results_lasso():
    lmb = 0.5
    model = LassoReg(poly_degrees)
    error, _, bias, variance = model.simple_regression(x, y, z, lmb, True)
    bias_variance(
        poly_degrees, error, bias, variance, N,
        title=r'Lasso regression with simple resampling for $\lambda = $' + f'{lmb}',
        fname=f'plots/Lasso/Lasso_bias_variance_lmb_{lmb}_{N}.pdf'
    )

    error, bias, variance = model.bootstrap(x, y, z, lambdas, N_boostraps)
    for i in range(0, N_lambdas, 2):
        bias_variance(
            poly_degrees, error[:, :, i], bias[:, :, i], variance[:, :, i], N,
            title=r'Lasso regression with bootstrapping for $\lambda = $' + f'{lambdas[i]:.2e}',
            fname=f'plots/Lasso/Lasso_BS_{i+1}_{N}.pdf'
        )
    error, error_sklearn = model.k_fold(xy, x, y, z, lambdas, k)
    deg_idx = 1
    lmb_idx = 1
    error_of_polydeg(
        poly_degrees, error[:, :, lmb_idx],
        title=r'Lasso regression with k-foldning for $\lambda = $' + f'{lambdas[lmb_idx]:.2e}',
        fname=f'plots/Lasso/Lasso_k-fold_lmb_{lambdas[lmb_idx]:.2e}_{N}.pdf'
    )

    owncode_vs_sklearn(poly_degrees, error[:, :, lmb_idx], error_sklearn[:, lmb_idx],
        title='Lasso regression with k-foldning for $\lambda = $' + f'{lambdas[lmb_idx]:.2e}',
        fname=f'plots/Lasso/Lasso_k-fold_sklearn_lmb_{lambdas[lmb_idx]:.2e}_{N}.pdf'
    )

    error_of_lambda(
        lambdas, error[:, deg_idx],
        title=f'Lasso regression with k-foldning for polynomial degree {poly_degrees[deg_idx]}',
        fname=f'plots/Lasso/Lasso_k-fold_lambda_deg_{poly_degrees[deg_idx]:.2e}_{N}.pdf'
    )

#produce_results_OLS()
produce_results_ridge()
produce_results_lasso()
