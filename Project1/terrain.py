import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from regression_methods import OLSReg, RidgeReg, LassoReg
from plot import (MSE_R2, bias_variance_error, error_of_polydeg, owncode_vs_sklearn,
error_of_lambda, plot_terrain, model_terrain)

# Load the terrain
terrain = imread('map_data/SRTM_data_Norway_1.tif')

N = 500 #len(terrain[0])
poly_degrees = np.arange(1, 8)
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

z = np.array(terrain)

def model_terrain_Norway():
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    bbox = fig.bbox_inches.from_bounds(1, 1, 8, 6)
    ax.set_zticklabels([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(f'Terrain of Stavanger with {N}x{N} data points', fontsize=20)
    surf = ax.plot_surface(x_mesh, y_mesh, z, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # plt.savefig(f'plots/Terrain_Norway/Stavanger_N_{N}.pdf')
    plt.show()

# # Show the terrain
# plot_terrain(terrain)

def produce_results_OLS():
    deg_idx = 8
    model = OLSReg(poly_degrees)
    error, r2, bias, variance, X_array, beta_array = model.simple_regression(x_mesh, y_mesh, z, 0, True)
    X = X_array[deg_idx]
    beta = beta_array[deg_idx]
    model_terrain(X, x_mesh, y_mesh, beta, N,
    title = f'OLS regression with p={poly_degrees[deg_idx]}',
    fname = f'plots/Terrain_Norway/OLS_surf_p{poly_degrees[deg_idx]}_N{N}.pdf'
    )
    MSE_R2(
        poly_degrees, error, r2,
        title=f'{N}x{N} data points',
        fname=f'plots/Terrain_Norway/OLS_MSE_R2score{N}.pdf'
    )

    error, bias, variance = model.bootstrap(x_mesh, y_mesh, z, [0], N_boostraps)
    bias_variance_error(
        poly_degrees, error, bias, variance, N,
        title='OLS regression with bootstrapping',
        fname=f'plots/Terrain_Norway/OLS_BiasVariance_BS_{N}.pdf'
    )

    error, error_sklearn = model.k_fold(xy, x, y, z, [0], k)
    error_of_polydeg(
        poly_degrees, error,
        title=f'OLS regression with k-foldning',
        fname=f'plots/Terrain_Norway/OLS_k-fold{N}.pdf'
    )

def produce_results_Ridge():
    model = RidgeReg(poly_degrees)
    lmb = 0.1
    error, _, bias, variance = model.simple_regression(x_mesh, y_mesh, z, lmb, True)
    test_vs_train(
        poly_degrees, error, N,
        title=f'OLS regression without resampling for {N}x{N} datapoints',
        fname=f'plots/Terrain_Norway/OLS_Test_vs_train_{N}.pdf'
    )
    lmb = 0.1
    error, bias, variance = model.bootstrap(x, y, z, lmb, N_boostraps)
    bias_variance(
        poly_degrees, error, bias, variance, N,
        title='OLS regression with bootstrapping',
        fname=f'plots/Terrain_Norway/OLS_BiasVariance_BS_lmb_{lmb}_{N}.pdf'
    )

    error, _ = model.k_fold(xy, x_mesh, y_mesh, z, lambdas, k)
    deg_idx = 8
    error_of_lambda(
        lambdas, error[:, deg_idx],
        title=f'Ridge regression with k-foldning for polynomial degree {poly_degrees[deg_idx]}',
        fname=f'plots/Terrain_Norway/Ridge_k-fold_lambda_deg_{deg_idx}_{N}.pdf'
    )
    lmb_idx = 3
    error_of_polydeg(
        poly_degrees, error[:, :, lmb_idx],
        title=r'Ridge regression with k-foldning for $\lambda$ = $' + f'{lambdas[lmb_idx]:.2e}',
        fname=f'plots/Terrain_Norway/Ridge_k-fold_lmb_{lambdas[lmb_idx]:.2e}_{N}.pdf'
    )


def produce_results_lasso():
    model = LassoReg(poly_degrees)
    lmb = 0.1
    error, _, bias, variance = model.simple_regression(x_mesh, y_mesh, z, lmb, True)


    error, _ = model.k_fold(xy, x_mesh, y_mesh, z, lambdas, k)
    deg_idx = 8
    error_of_lambda(
        lambdas, error[:, deg_idx],
        title=f'Lasso regression with k-foldning for polynomial degree {poly_degrees[deg_idx]}',
        fname=f'plots/Terrain_Norway/Lasso_k-fold_lambda_deg_{poly_degrees[deg_idx]}_{N}.pdf'
    )
    lmb_idx = 3
    error_of_polydeg(
        poly_degrees, error[:, :, lmb_idx],
        title=r'Lasso regression with k-foldning for $\lambda = $' + f'{lambdas[lmb_idx]:.2e}',
        fname=f'plots/Terrain_Norway/Lasso_k-fold_lmb_{lambdas[lmb_idx]:.2e}_{N}.pdf'
    )


# model_terrain_Norway()
produce_results_OLS()
# produce_results_Ridge()
# produce_results_lasso()


