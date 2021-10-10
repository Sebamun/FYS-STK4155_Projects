import numpy as np
from imageio import imread
from regression_methods import OLSReg, RidgeReg, LassoReg
from plot import (MSE_R2, bias_variance, error_of_polydeg, owncode_vs_sklearn,
error_of_lambda, plot_terrain)

# Load the terrain
terrain = imread('map_data/SRTM_data_Norway_1.tif')

N = 1000 #len(terrain[0])
poly_degrees = np.arange(1, 10)
N_boostraps = 100
k = 5
N_lambdas = 9
lambdas = np.logspace(-4, 1, N_lambdas)
terrain = terrain[:N,:N]

# Creates mesh of image pixels
x = np.linspace(0, 1, np.shape(terrain)[0])
y = np.linspace(0, 1, np.shape(terrain)[1])

xy = np.zeros((N,2))
xy[:,0] = x
xy[:,1] = y

x_mesh, y_mesh = np.meshgrid(x,y)

z = terrain

# # Show the terrain
# plot_terrain(terrain)

def produce_results_OLS():
    model = OLSReg(poly_degrees)
    error, r2, bias, variance = model.simple_regression(x_mesh, y_mesh, z, 0, True)
    MSE_R2(
        poly_degrees, error, r2,
        title='OLS regression with simple resampling'
        fname=f'plots/exercise6_OLS_MSE_R2score{N}.pdf'
    )
    bias_variance(
        poly_degrees, error, bias, variance,
        title='OLS regression with simple resampling',
        fname=f'plots/exercise6_OLS_bias_variance{N}.pdf'
    )

    error, error_sklearn = model.k_fold(xy, x, y, z, [0], k)
    error_of_polydeg(
        poly_degrees, error,
        title=f'OLS regression with k-foldning',
        fname=f'plots/exercise6_OLS_k-fold{N}.pdf'
    )

def produce_results_Ridge():
    model = RidgeReg(poly_degrees)
    lmb = 0.1
    error, _, bias, variance = model.simple_regression(x_mesh, y_mesh, z, lmb, True)
    bias_variance(
        poly_degrees, error, bias, variance,
        title=r'Ridge regression with simple resampling for $\lambda = $' + f'{lmb}',
        fname=f'plots/exercise6_Ridge_bias_variance_lmb_{lmb}_{N}.pdf'
    )

    error, _ = model.k_fold(xy, x_mesh, y_mesh, z, lambdas, k)
    deg_idx = 10
    error_of_lambda(
        lambdas, error[:, deg_idx],
        title=f'Ridge regression with k-foldning for polynomial degree {poly_degrees[deg_idx]}',
        fname=f'plots/exercise6_Ridge_k-fold_lambda_deg_{deg_idx}_{N}.pdf'
    )
    lmb_idx = 3
    error_of_polydeg(
        poly_degrees, error[:, :, lmb_idx],
        title=r'Ridge regression with k-foldning for $\lambda$ = $' + f'{lambdas[lmb_idx]:.2e}',
        fname=f'plots/exercise6_Ridge_k-fold_lmb_{lambdas[lmb_idx]:.2e}_{N}.pdf'
    )

def produce_results_lasso():
    model = LassoReg(poly_degrees)
    lmb = 0.1
    error, _, bias, variance = model.simple_regression(x_mesh, y_mesh, z, lmb, True)
    bias_variance(
        poly_degrees, error, bias, variance,
        title=r'Lasso simple resampling with $\lambda = $' + f'{lmb}',
        fname=f'plots/exercise6_Lasso_bias_variance_lmb{lmb}_{N}.pdf'
    )

    error, _ = model.k_fold(xy, x_mesh, y_mesh, z, lambdas, k)
    deg_idx = 10
    error_of_lambda(
        lambdas, error[:, deg_idx],
        title=f'Lasso regression with k-foldning for polynomial degree {poly_degrees[deg_idx]}',
        fname=f'plots/exercise6_Lasso_k-fold_lambda_deg_{poly_degrees[deg_idx]}_{N}.pdf'
    )
    lmb_idx = 3
    error_of_polydeg(
        poly_degrees, error[:, :, lmb_idx],
        title=r'Lasso regression with k-foldning for $\lambda = $' + f'{lambdas[lmb_idx]:.2e}',
        fname=f'plots/exercise6_Lasso_k-fold_lmb_{lambdas[lmb_idx]:.2e}_{N}.pdf'
    )

# produce_results_OLS()
# produce_results_Ridge()
produce_results_lasso()
