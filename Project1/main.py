import numpy as np
import matplotlib.pyplot as plt
from regression_methods import OLSReg, RidgeReg, LassoReg
from common import FrankeFunction, compute_means_over_lambda
from plot import (MSE_R2, bias_variance, error_of_polydeg, owncode_vs_sklearn, test_vs_train,
error_of_lambda, variance_of_lambda, bias_variance_of_lambda, bias_variance_poly, bias_variance_comp)

N = 20
poly_degrees = np.arange(1, 14)
N_boostraps = 20
k = 5
N_lambdas = 20
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
    mean_bias, mean_variance = compute_means_over_lambda(bias, variance, N_lambdas)
    bias_variance_poly(
        poly_degrees, error, mean_bias, mean_variance,
        title=r'Bias Variance trend, taken as mean over all $\lambda$, ' + f'{N}x{N}',
        fname=f'plots/Ridge/Ridge_BiasVariance_BS_mean_{N}.pdf'
    )
    deg_idx = 6
    lmb_idx = 10
    bias_variance_poly(
        poly_degrees, error[:, :, lmb_idx], bias[:, :, lmb_idx], variance[:, :, lmb_idx],
        title=r'Bias Variance, Ridge, bootstrap, $\lambda$=' + f'{lmb_idx}' + f', {N}x{N}',
        fname=f'plots/Ridge/Ridge_BiasVariance_BS{N}.pdf'
    )
    bias_variance_of_lambda(
        lambdas, variance[:, deg_idx], bias[:, deg_idx],
        title=f'Ridge bias-variance, {N}x{N}, p={poly_degrees[deg_idx]}',
        fname=f'plots/Ridge/Ridge_BiasVariance_BS_lambda_{poly_degrees[deg_idx]:.2e}_{N}.pdf'
    )

    error, error_sklearn = model.k_fold(xy, x, y, z, lambdas, k)
    variance_of_lambda(
        lambdas, variance[:, deg_idx],
        title=f'Ridge regression with k-foldning for polynomial degree {poly_degrees[deg_idx]}',
        fname=f'plots/Ridge/Ridge_variance_k-fold_lambda_deg_{poly_degrees[deg_idx]:.2e}_{N}.pdf'
    )
    error_of_polydeg(
        poly_degrees, error[:, :, lmb_idx],
        title=r'Ridge regression with k-foldning for $\lambda = $' + f'{lambdas[lmb_idx]:.2e}',
        fname=f'plots/Ridge/Ridge_k-fold_lmb_{lambdas[lmb_idx]:.2e}_{N}.pdf'
    )
    owncode_vs_sklearn(poly_degrees, error[:, :, lmb_idx], error_sklearn[:, lmb_idx],
        title='Ridge regression with k-foldning for $\lambda = $' + f'{lambdas[lmb_idx]:.2e}',
        fname=f'plots/Ridge/Ridge_k-fold_sklearn_lmb_{lambdas[lmb_idx]:.2e}_{N}.pdf')

    error_of_lambda(
        lambdas, error[:, deg_idx],
        title=f'Ridge regression with k-foldning for polynomial degree {poly_degrees[deg_idx]}',
        fname=f'plots/Ridge/Ridge_k-fold_lambda_deg_{poly_degrees[deg_idx]:.2e}_{N}.pdf'
    )

    l_idx0 = 0
    l_idx1 = 5 # This gives the low lambda value.
    l_idx2 = 10
    l_idx3 = 19 # This gives the high lambda value.
    l0 = lambdas[l_idx0]
    l1 = lambdas[l_idx1] # Low lambda value.
    l2 = lambdas[l_idx2] # High lambda value.
    l3 = lambdas[l_idx3] # High lambda value.
    bias_variance_comp(
        poly_degrees,
        bias[:, :, l_idx0], variance[:, :, l_idx0],
        bias[:, :, l_idx1], variance[:, :, l_idx1],
        bias[:, :, l_idx2], variance[:, :, l_idx2],
        bias[:, :, l_idx3], variance[:, :, l_idx3],
        l0, l1, l2, l3,
        fname=f'plots/Ridge/Ridge_comp_lambda.pdf'
    )

    #l_idx1 = 0
    #l_idx2 = 5 # This gives the low lambda value.
    #l_idx3 = 10
    #l_idx4 = 19 # This gives the high lambda value.
    #l1 = lambdas[l_idx1] # Low lambda value.
    #l2 = lambdas[l_idx2] # High lambda value.
    #l3 = lambdas[l_idx3]
    #l4 = lambdas[l_idx4]
    #bias_variance_comp(
    #    poly_degrees, bias[:, :, l_idx1], variance[:, :, l_idx1], bias[:, :, l_idx2], variance[:, :, l_idx2],
    #    bias[:, :, l_idx3], variance[:, :, l_idx3], bias[:, :, l_idx4], variance[:, :, l_idx4],l1,l2,l3,l4,
    #    fname=f'plots/Ridge/Ridge_comp_lambda.pdf'

    #)

""" LASSO; SIMPLE, BOOTSTRAP, K-FOLD """
def produce_results_lasso():
    model = LassoReg(poly_degrees)
    '''
    lmb = 0.5
    error, _, bias, variance = model.simple_regression(x, y, z, lmb, True)
    bias_variance(
        poly_degrees, error, bias, variance, N,
        title=r'Lasso regression with simple resampling for $\lambda = $' + f'{lmb}',
        fname=f'plots/Lasso/Lasso_bias_variance_lmb_{lmb}_{N}.pdf'
    )
    '''
    error, bias, variance = model.bootstrap(x, y, z, lambdas, N_boostraps)
    mean_bias, mean_variance = compute_means_over_lambda(bias, variance, N_lambdas)
    bias_variance_poly(
        poly_degrees, error, mean_bias, mean_variance,
        title=r'Bias Variance trend, taken as mean over all $\lambda$, Lasso, )' + f'{N}x{N}',
        fname=f'plots/Lasso/Lasso_BiasVariance_BS_mean_{N}.pdf'
    )
    error, error_sklearn = model.k_fold(xy, x, y, z, lambdas, k)
    deg_idx = 6
    lmb_idx = 10
    bias_variance_poly(
        poly_degrees, error[:, :, lmb_idx], bias[:, :, lmb_idx], variance[:, :, lmb_idx],
        title=r'Bias Variance, Lasso, kfold, $\lambda$=)' + f'{lmb_idx}' + f', {N}x{N}',
        fname=f'plots/Lasso/Lasso_BiasVariance_kfold_{N}.pdf'
    )
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
    l_idx1 = 0
    l_idx2 = 5 # This gives the low lambda value.
    l_idx3 = 10
    l_idx4 = 19 # This gives the high lambda value.
    l1 = lambdas[l_idx1] # Low lambda value.
    l2 = lambdas[l_idx2] # High lambda value.
    l3 = lambdas[l_idx3]
    l4 = lambdas[l_idx4]
    bias_variance_comp(
        poly_degrees, bias[:, :, l_idx1], variance[:, :, l_idx1], bias[:, :, l_idx2], variance[:, :, l_idx2],
        bias[:, :, l_idx3], variance[:, :, l_idx3], bias[:, :, l_idx4], variance[:, :, l_idx4],l1, l2,l3,l4,
        fname=f'plots/Lasso/Lasso_comp_lambda.pdf'
    )
#produce_results_OLS()
produce_results_ridge()
#produce_results_lasso()
