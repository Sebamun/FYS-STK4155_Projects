import numpy as np
from regression_methods import OLSReg, RidgeReg, LassoReg
from common import FrankeFunction, compute_means_over_lambda
from plot import (MSE_R2, bias_variance_error, error_of_polydeg, owncode_vs_sklearn,
error_of_lambda, model_terrain, confidence_intervall, bias_variance_compare, error_of_polydeg_compare)

N = 20
poly_degrees = np.arange(1, 10)
N_boostraps = 20
k = 10
N_lambdas = 20
lambdas = np.logspace(-4, 1, N_lambdas)

np.random.seed(2018)

x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)

xy = np.zeros((N,2))
xy[:,0] = x
xy[:,1] = y

# Making meshgrid of datapoints and compute Franke's function
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y)

""" Terrain """
def produce_terrain():
    deg_idx = 7
    model = OLSReg(poly_degrees)
    *_, X_array, beta_array = model.simple_regression(x, y, z, 0, False)
    X = X_array[deg_idx]
    beta = beta_array[deg_idx]
    model_terrain(
        X, x, y, beta, N,
        title = f'OLS regression with p={poly_degrees[deg_idx]}',
        fname = f'plots/OLS/OLS_surf_p{poly_degrees[deg_idx]}_N{N}.pdf'
    )

    deg_idx = 7
    model = RidgeReg(poly_degrees)
    *_, X_array, beta_array = model.simple_regression(x, y, z, 4.28e-2, False)
    X = X_array[deg_idx]
    beta = beta_array[deg_idx]
    model_terrain(
        X, x, y, beta, N,
        title = f'Ridge regression with p={poly_degrees[deg_idx]}',
        fname = f'plots/Ridge/Ridge_surf_p{poly_degrees[deg_idx]}_N{N}.pdf'
    )

""" OLS; SIMPLE, BOOTSTRAP, K-FOLD """
def produce_results_OLS():
    model = OLSReg(poly_degrees)

    # Without resampling
    error, r2, bias, variance, X_array, beta_array = model.simple_regression(x, y, z, 0, True)
    deg_idx = 4
    confidence_intervall(
        X_array[deg_idx], beta_array[deg_idx], N,
        title=r'$\beta$ values with corresponding confidence interval of 95$\%$,' + f'p={poly_degrees[deg_idx]}, n={N}',
        fname=f'plots/OLS/OLSconf_plot_p{poly_degrees[deg_idx]}_N{N}.png'
    )
    MSE_R2(
        poly_degrees, error, r2,
        title=f'{N}x{N} data points',
        fname=f'plots/OLS/OLS_MSE_R2score{N}.pdf'
    )

    # Bootstrapping
    error, bias, variance = model.bootstrap(x, y, z, [0], N_boostraps)
    bias_variance_error(
        poly_degrees, error, bias, variance, N,
        title='OLS regression with bootstrapping',
        fname=f'plots/OLS/OLS_BiasVariance_BS_{N}.pdf'
    )

    # K-folding
    error, error_sklearn = model.k_fold(xy, x, y, z, [0], k)
    error_of_polydeg(
        poly_degrees, error,
        title=f'OLS regression with k-foldning for k = {k}',
        fname=f'plots/OLS/OLS_kfold_{k}_MSE{N}.pdf'
    )
    owncode_vs_sklearn(
        poly_degrees, error[:, :, 0], error_sklearn[:, 0],
        title=f'K-foldning with OLS and sklearn for k = {k}',
        fname=f'plots/OLS/OLS_k-fold_sklearn_{k}_{N}.pdf'
    )

""" RIDGE; SIMPLE, BOOTSTRAP, K-FOLD """
def produce_results_ridge():
    model = RidgeReg(poly_degrees)

    # Bootstrapping
    error, bias, variance = model.bootstrap(x, y, z, lambdas, N_boostraps)
    deg_idx = 8
    lmb_idx = 10
    error_of_polydeg(
        poly_degrees, error[:, :, lmb_idx],
        title='Ridge regression with bootstrapping for $\lambda = $' + f'{lambdas[lmb_idx]:.2e}',
        fname=f'plots/Ridge/Ridge_bootstrap_lmb_{lambdas[lmb_idx]:.2e}_{N}.pdf'
    )

    l_idx0 = 0
    l_idx1 = 5
    l_idx2 = 10
    l_idx3 = 19
    l0 = lambdas[l_idx0] # Low lambda value.
    l1 = lambdas[l_idx1]
    l2 = lambdas[l_idx2]
    l3 = lambdas[l_idx3] # High lambda value.
    bias_variance_compare(
        poly_degrees,
        bias[:, :, l_idx0], variance[:, :, l_idx0],
        bias[:, :, l_idx1], variance[:, :, l_idx1],
        bias[:, :, l_idx2], variance[:, :, l_idx2],
        bias[:, :, l_idx3], variance[:, :, l_idx3],
        l0, l1, l2, l3,
        fname=f'plots/Ridge/Ridge_comp_lambda.pdf'
    )

    # K-folding
    error, error_sklearn = model.k_fold(xy, x, y, z, lambdas, k)
    l_idx0 = 0
    l_idx1 = 5
    l_idx2 = 10
    l_idx3 = 19
    l0 = lambdas[l_idx0] # Low lambda value.
    l1 = lambdas[l_idx1]
    l2 = lambdas[l_idx2]
    l3 = lambdas[l_idx3] # High lambda value.
    error_of_polydeg_compare(
        poly_degrees, error[:, :, l_idx0], error[:, :, l_idx1], error[:, :, l_idx2], error[:, :, l_idx3],
        l0, l1, l2, l3,
        title=f'Ridge regression with k-foldning for k = {k}',
        fname=f'plots/Ridge/Ridge_comp_lambda_k-fold_{k}_{N}.pdf'
    )
    deg_idx = 3
    lmb_idx = 5
    owncode_vs_sklearn(poly_degrees, error[:, :, lmb_idx], error_sklearn[:, lmb_idx],
        title='Ridge regression with k-foldning for $\lambda = $' + f'{lambdas[lmb_idx]:.2e}',
        fname=f'plots/Ridge/Ridge_k-fold_sklearn_{k}_lmb_{lambdas[lmb_idx]:.2e}_{N}.pdf'
    )
    error_of_lambda(
        lambdas, error[:, deg_idx],
        title=f'Ridge regression with k-foldning for polynomial degree {poly_degrees[deg_idx]}',
        fname=f'plots/Ridge/Ridge_k-fold_{k}_lambda_deg_{poly_degrees[deg_idx]:.2e}_{N}.pdf'
    )

""" LASSO; SIMPLE, BOOTSTRAP, K-FOLD """
def produce_results_lasso():
    model = LassoReg(poly_degrees)
    error, r2, bias, variance, X_array, beta_array = model.simple_regression(x, y, z, 0, True)
    lmb_idx = 10
    # The following function call gives ConvergenceWarning which might be due to noice in the FrankeFunction
    error_of_polydeg(
        poly_degrees, error,
        title='Lasso regression without resampling for $\lambda = $' + f'{lambdas[lmb_idx]:.2e}',
        fname=f'plots/Lasso/Lasso_lmb_{lambdas[lmb_idx]:.2e}_{N}.pdf'
    )

    # Bootstrapping
    error, bias, variance = model.bootstrap(x, y, z, lambdas, N_boostraps)
    lmb_idx = 10
    error_of_polydeg(
        poly_degrees, error[:, :, lmb_idx],
        title='Lasso regression with boothstrapping for $\lambda = $' + f'{lambdas[lmb_idx]:.2e}',
        fname=f'plots/Lasso/Lasso_bootstrap_lmb_{lambdas[lmb_idx]:.2e}_{N}.pdf'
    )
    l_idx0 = 0
    l_idx1 = 5
    l_idx2 = 10
    l_idx3 = 19
    l0 = lambdas[l_idx0] # Low lambda value.
    l1 = lambdas[l_idx1]
    l2 = lambdas[l_idx2]
    l3 = lambdas[l_idx3] # High lambda value.
    bias_variance_compare(
        poly_degrees,
        bias[:, :, l_idx0], variance[:, :, l_idx0],
        bias[:, :, l_idx1], variance[:, :, l_idx1],
        bias[:, :, l_idx2], variance[:, :, l_idx2],
        bias[:, :, l_idx3], variance[:, :, l_idx3],
        l0, l1, l2, l3,
        fname=f'plots/Lasso/Lasso_comp_lambda.pdf'
    )

    # K-folding
    error, error_sklearn = model.k_fold(xy, x, y, z, lambdas, k)
    l_idx0 = 0
    l_idx1 = 5
    l_idx2 = 10
    l_idx3 = 19
    l0 = lambdas[l_idx0] # Low lambda value.
    l1 = lambdas[l_idx1]
    l2 = lambdas[l_idx2]
    l3 = lambdas[l_idx3] # High lambda value.
    error_of_polydeg_compare(
        poly_degrees, error[:, :, l_idx0], error[:, :, l_idx1], error[:, :, l_idx2], error[:, :, l_idx3],
        l0, l1, l2, l3,
        title=f'Lasso regression with k-foldning for k = {k}',
        fname=f'plots/Lasso/Lasso_comp_lambda_k-fold_{k}_{N}.pdf'
    )
    deg_idx = 6
    lmb_idx = 5
    error_of_polydeg(
        poly_degrees, error[:, :, lmb_idx],
        title=r'Lasso regression with k-foldning for $\lambda = $' + f'{lambdas[lmb_idx]:.2e}',
        fname=f'plots/Lasso/Lasso_k-fold_{k}_lmb_{lambdas[lmb_idx]:.2e}_{N}.pdf'
    )
    owncode_vs_sklearn(poly_degrees, error[:, :, lmb_idx], error_sklearn[:, lmb_idx],
        title='Lasso regression with k-foldning for $\lambda = $' + f'{lambdas[lmb_idx]:.2e}',
        fname=f'plots/Lasso/Lasso_k-fold_sklearn_{k}_lmb_{lambdas[lmb_idx]:.2e}_{N}.pdf'
    )
    error_of_lambda(
        lambdas, error[:, deg_idx],
        title=f'Lasso regression with k-foldning for polynomial degree {poly_degrees[deg_idx]}',
        fname=f'plots/Lasso/Lasso_k-fold_{k}_lambda_deg_{poly_degrees[deg_idx]:.2e}_{N}.pdf'
    )

# produce_terrain()
# produce_results_OLS()
# produce_results_ridge()
# produce_results_lasso()
