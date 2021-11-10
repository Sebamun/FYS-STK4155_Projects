import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

def confidence_intervall(X, beta, N, title, fname):
    fig, ax = plt.subplots()
    diag_ele = np.zeros(len(beta))
    X_i = np.linalg.pinv(X.T@X)
    for i in range(len(beta)):
        diag_ele[i] = X_i[i,i]
    var_error = np.sqrt(np.var(np.random.normal(0,0.1,N)))
    var_beta = var_error*diag_ele
    betas = np.arange(1, len(beta) + 1)
    CI_u = beta + 1.96*var_beta/np.sqrt(N)
    CI_l = beta - 1.96*var_beta/np.sqrt(N)
    yerr = 2*var_beta/np.sqrt(N)
    plt.fill_between(betas, CI_u, CI_l, alpha = 0.9)
    plt.scatter(betas, beta, marker = 'x', color = 'red', label = 'beta')
    plt.title(title, fontsize=18)
    plt.xlabel(r'$\beta$', fontsize=18)
    plt.xticks(betas)
    plt.ylabel(r'$\beta$ values', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.legend()
    plt.savefig(fname)

def MSE_R2(poly_degrees, MSE, R2, title, fname):
    step = 3
    poly_degrees_new = np.arange(poly_degrees[0], len(poly_degrees), step)
    fig, axes = plt.subplots(1, 2)
    ax = axes[0]
    ax.set_xticks(poly_degrees_new)
    ax.tick_params(axis='both', which='major', labelsize=18)
    fig.suptitle(title, fontsize=20)
    ax.set_title('Mean Square Error', fontsize=18)
    ax.plot(poly_degrees, MSE[0], label="Training data")
    ax.plot(poly_degrees, MSE[1], label="Test data")
    ax.set_xlabel('Polynomial Degree', fontsize=18)
    ax.set_ylabel('MSE', fontsize=18)
    ax.legend(fontsize=18)

    ax = axes[1]
    ax.set_xticks(poly_degrees_new)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_title('R2 score', fontsize=18)
    ax.plot(poly_degrees, R2[0], label="Training data")
    ax.plot(poly_degrees, R2[1], label="Test data")
    ax.set_xlabel('Polynomial Degree', fontsize=18)
    ax.set_ylabel('R2', fontsize=18)
    ax.legend(fontsize=18)
    fig.savefig(fname)
    plt.close(fig)

def bias_variance_error(poly_degrees, MSE, bias, variance, N, title, fname):
    poly_degrees_new = np.arange(1, len(poly_degrees), 2)
    fig, ax = plt.subplots()
    ax.set_title(title, fontsize=20)
    ax.set_xticks(poly_degrees_new)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_title(f'Test data, {N}x{N} datapoints', fontsize=18)
    ax.plot(poly_degrees, MSE[1], label='MSE')
    ax.plot(poly_degrees, bias[1], label='Bias')
    ax.plot(poly_degrees, variance[1], label='Variance')
    ax.set_xlabel('Polynomial Degree', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(fontsize=18)
    fig.savefig(fname)
    plt.close(fig)

# def bias_variance(poly_degrees, MSE, bias, variance, title, fname):
#     poly_degrees_new = np.arange(1, len(poly_degrees), 2)
#     fig, axes = plt.subplots(1, 2)
#     fig.suptitle(title, fontsize=20)
#     ax = axes[0]
#     ax.set_xticks(poly_degrees_new)
#     ax.tick_params(axis='both', which='major', labelsize=18)
#     ax.plot(poly_degrees, variance[1], label='Variance')
#     ax.set_xlabel('Polynomial Degree', fontsize=18)
#     ax.legend(fontsize=18)
#
#     ax = axes[1]
#     ax.set_xticks(poly_degrees_new)
#     ax.tick_params(axis='both', which='major', labelsize=18)
#     ax.plot(poly_degrees, bias[1], label='Bias')
#     ax.set_xlabel('Polynomial Degree', fontsize=18)
#     ax.legend(fontsize=18)
#     fig.savefig(fname)
#     plt.close(fig)

# def variance_of_lambda(lambdas, var_lmbd, title, fname):
#     fig, ax = plt.subplots()
#     ax.set_title(title, fontsize=20)
#     ax.plot(np.log10(lambdas), var_lmbd[0], label = 'Variance, train data')
#     ax.plot(np.log10(lambdas), var_lmbd[1], label = 'Variance, test data')
#     ax.set_xlabel(r'$\log_{10}(\lambda)$', fontsize=18)
#     ax.set_ylabel('Variance', fontsize=18)
#     ax.legend(fontsize=18)
#     fig.savefig(fname)
#     plt.close(fig)

# def bias_variance_of_lambda(lambdas, var, bias, title, fname):
#     var_min = np.argmin(var[1])
#     fig, ax = plt.subplots(1,2)
#     fig.suptitle(title, fontsize=20)
#     ax[0].set_title('Variance')
#     ax[0].plot(np.log10(lambdas), var[1], label = 'Variance')
#     ax[0].scatter(np.log10(lambdas[var_min]), var[1][var_min], label = r'$\lambda$ =' + f'{lambdas[var_min]:.2f}')
#     ax[1].set_title('Bias')
#     ax[1].plot(np.log10(lambdas), bias[1], label = 'Bias')
#     ax[1].scatter(np.log10(lambdas[var_min]), bias[1][var_min], label = r'$\lambda$ =' + f'{lambdas[var_min]:.2f}')
#     ax[0].set_xlabel(r'$\log_{10}(\lambda)$', fontsize=18)
#     ax[1].set_xlabel(r'$\log_{10}(\lambda)$', fontsize=18)
#     ax[0].legend(fontsize=18)
#     ax[1].legend(fontsize=18)
#     fig.savefig(fname)
#     plt.close(fig)

def error_of_lambda(lambdas, error_lmbd, title, fname):
    error_min = np.argmin(error_lmbd[1])
    fig, ax = plt.subplots()
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_title(title, fontsize=20)
    ax.plot(np.log10(lambdas), error_lmbd[0], label = 'MSE, train data')
    ax.plot(np.log10(lambdas), error_lmbd[1], label = 'MSE, test data')
    ax.scatter(np.log10(lambdas[error_min]), error_lmbd[1][error_min], label = r'$\lambda$ =' + f'{lambdas[error_min]:.4f}')
    ax.set_xlabel(r'$\log_{10}(\lambda)$', fontsize=18)
    ax.set_ylabel('MSE', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(fontsize=18)
    fig.savefig(fname)
    plt.close(fig)

def error_of_polydeg(poly_degrees, MSE, title, fname):
    poly_degrees_new = np.arange(1, len(poly_degrees), 2)
    fig, ax = plt.subplots()
    ax.set_xticks(poly_degrees_new)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_title(title, fontsize=20)
    ax.plot(poly_degrees, MSE[0], label = 'MSE, train data')
    ax.plot(poly_degrees, MSE[1], label = 'MSE, test data')
    ax.set_xlabel('Polynomial Degree', fontsize=18)
    ax.set_ylabel('MSE', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(fontsize=18)
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)

def owncode_vs_sklearn(poly_degrees, MSE, MSE_sklearn, title, fname):
    poly_degrees_new = np.arange(1, len(poly_degrees), 2)
    fig, ax = plt.subplots()
    ax.set_xticks(poly_degrees_new)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_title(title + ' with test data', fontsize=20)
    ax.plot(poly_degrees, MSE_sklearn, label = 'MSE, sklearn')
    ax.plot(poly_degrees, MSE[1], 'r--', label = 'MSE, own code')
    ax.set_xlabel('Polynomial Degree', fontsize=18)
    ax.set_ylabel('MSE', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(fontsize=18)
    fig.savefig(fname)
    plt.close(fig)

def error_of_polydeg_compare(poly_degrees, MSE_0, MSE_1, MSE_2, MSE_3, l_idx0, l_idx1, l_idx2, l_idx3, title, fname):
    poly_degrees_new = np.arange(1, len(poly_degrees), 2)
    fig, ax = plt.subplots()
    ax.set_xticks(poly_degrees_new)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_title(title, fontsize=20)
    ax.plot(poly_degrees, np.log10(MSE_0[1]), label=f'$\lambda$={l_idx0:.2e}')
    ax.plot(poly_degrees, np.log10(MSE_1[1]), label=f'$\lambda$={l_idx1:.2e}')
    ax.plot(poly_degrees, np.log10(MSE_2[1]), label=f'$\lambda$={l_idx2:.2e}')
    ax.plot(poly_degrees, np.log10(MSE_3[1]), label=f'$\lambda$={l_idx3:.2e}')
    ax.set_xlabel('Polynomial Degree', fontsize=18)
    ax.set_ylabel('log10[MSE]', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(fontsize=18)
    fig.savefig(fname)
    plt.close(fig)

def bias_variance_compare(poly_degrees, bias_0, variance_0, bias_1, variance_1, bias_2, variance_2, bias_3, variance_3, l_idx0, l_idx1, l_idx2, l_idx3, fname):
    poly_degrees_new = np.arange(0, len(poly_degrees), 2)
    fig, axes = plt.subplots(1, 2)
    fig.suptitle('Bias and variance given $\lambda$', fontsize=20)
    ax = axes[0]
    ax.set_xticks(poly_degrees_new)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_title("Variance", fontsize=18)
    ax.plot(poly_degrees, variance_0[1], label=f'$\lambda$={l_idx0:.2e}')
    ax.plot(poly_degrees, variance_1[1], label=f'$\lambda$={l_idx1:.2e}')
    ax.plot(poly_degrees, variance_2[1], label=f'$\lambda$={l_idx2:.2e}')
    ax.plot(poly_degrees, variance_3[1], color='orange', label=f'$\lambda$={l_idx3:.2e}')
    ax.set_xlabel('Polynomial Degree', fontsize=18)
    ax.legend(fontsize=18)

    ax = axes[1]
    ax.set_xticks(poly_degrees_new)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_title("Bias", fontsize=18)
    ax.plot(poly_degrees, bias_0[1], label=f'$\lambda$={l_idx0:.2e}')
    ax.plot(poly_degrees, bias_1[1], label=f'$\lambda$={l_idx1:.2e}')
    ax.plot(poly_degrees, bias_2[1], label=f'$\lambda$={l_idx2:.2e}')
    ax.plot(poly_degrees, bias_3[1], color='orange', label=f'$\lambda$={l_idx3:.2e}')
    ax.set_xlabel('Polynomial Degree', fontsize=18)
    ax.legend(fontsize=18)
    plt.savefig(fname)

def model_terrain(X, x, y, beta, N, title, fname):
    z = X@beta
    z_model = np.reshape(z, (N, N))
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    bbox = fig.bbox_inches.from_bounds(1, 1, 8, 6)
    ax.set_zticklabels([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    surf = ax.plot_surface(x, y, z_model, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)
    ax.set_title(title, fontsize=25)
    ax.zaxis.set_major_locator(LinearLocator(10))
    plt.savefig(fname, bbox_inches='tight')
