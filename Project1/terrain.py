import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from regression_methods import OLSReg, RidgeReg, LassoReg
from plot import (MSE_R2, bias_variance_error, error_of_polydeg, owncode_vs_sklearn,
error_of_lambda, model_terrain, error_of_polydeg_compare)

# Load the terrain
terrain = imread('map_data/SRTM_data_Norway_1.tif')

# Set initial conditions.
N = 10
poly_degrees = np.arange(1, 20)
N_boostraps = 100
k = 10 # Initialize a KFold instance, number of splitted parts
N_lambdas = 20 # Initialize number of bootstrap iterations
lambdas = np.logspace(-4, 1, N_lambdas) # Initialize lambdas for Ridge and Lasso
terrain = terrain[:N,:N]

# Creates mesh of image pixels
x = np.linspace(0, 1, np.shape(terrain)[0])
y = np.linspace(0, 1, np.shape(terrain)[1])

xy = np.zeros((N,2))
xy[:,0] = x
xy[:,1] = y

# Making meshgrid of datapoints and compute the terrain
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
    plt.savefig(f'plots/Terrain_Norway/Stavanger_N_{N}.pdf')

""" OLS; SIMPLE, BOOTSTRAP, K-FOLD """
def produce_results_OLS():
    model = OLSReg(poly_degrees)

    # Without resampling
    error, r2, X_array, beta_array = model.simple_regression(x_mesh, y_mesh, z, 0, True)
    deg_idx = 8
    X = X_array[deg_idx]
    beta = beta_array[deg_idx]
    model_terrain(
        X, x_mesh, y_mesh, beta, N,
        title = f'OLS regression with p={poly_degrees[deg_idx]}',
        fname = f'plots/Terrain_Norway/OLS_surf_p{poly_degrees[deg_idx]}_N{N}.pdf'
    )
    MSE_R2(
        poly_degrees, error, r2,
        title=f'{N}x{N} data points',
        fname=f'plots/Terrain_Norway/OLS_MSE_R2score{N}.pdf'
    )

    # Bootstrapping
    error, bias, variance = model.bootstrap(x_mesh, y_mesh, z, [0], N_boostraps)
    bias_variance_error(
        poly_degrees, error, bias, variance, N,
        title='OLS regression with bootstrapping',
        fname=f'plots/Terrain_Norway/OLS_BiasVariance_BS_{N}.pdf'
    )

    # K-folding
    error, error_sklearn = model.k_fold(xy, x, y, z, [0], k)
    error_of_polydeg(
        poly_degrees, error,
        title=f'OLS regression with k-foldning',
        fname=f'plots/Terrain_Norway/OLS_k-fold{N}.pdf'
    )

""" RIDGE; K-FOLD """
def produce_results_Ridge():
    model = RidgeReg(poly_degrees)

    # K-folding
    error, _ = model.k_fold(xy, x_mesh, y_mesh, z, lambdas, k)
    deg_idx = 2
    error_of_lambda(
        lambdas, error[:, deg_idx],
        title=f'Ridge regression with k-foldning for polynomial degree {poly_degrees[deg_idx]}',
        fname=f'plots/Terrain_Norway/Ridge_k-fold_lambda_deg_{deg_idx}_{N}.pdf'
    )
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
        fname=f'plots/Terrain_Norway/Ridge_comp_lambda_k-fold_{k}_{N}.pdf'
    )

""" LASSO; K-FOLD """
def produce_results_lasso():
    model = LassoReg(poly_degrees)

    # K-folding
    error, _ = model.k_fold(xy, x_mesh, y_mesh, z, lambdas, k)
    deg_idx = 1
    error_of_lambda(
        lambdas, error[:, deg_idx],
        title=f'Lasso regression with k-foldning for polynomial degree {poly_degrees[deg_idx]}',
        fname=f'plots/Terrain_Norway/Lasso_k-fold_lambda_deg_{deg_idx}_{N}.pdf'
    )
    l_idx0 = 0
    l_idx1 = 5
    l_idx2 = 10
    l_idx3 = 15
    l0 = lambdas[l_idx0] # Low lambda value.
    l1 = lambdas[l_idx1]
    l2 = lambdas[l_idx2]
    l3 = lambdas[l_idx3] # High lambda value.
    error_of_polydeg_compare(
        poly_degrees, error[:, :, l_idx0], error[:, :, l_idx1], error[:, :, l_idx2], error[:, :, l_idx3],
        l0, l1, l2, l3,
        title=f'Lasso regression with k-foldning for k = {k}',
        fname=f'plots/Terrain_Norway/Lasso_comp2_lambda_k-fold_{k}_{N}.pdf'
    )

model_terrain_Norway()
produce_results_OLS()
produce_results_Ridge()
produce_results_lasso()




