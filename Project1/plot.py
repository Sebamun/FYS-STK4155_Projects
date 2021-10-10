import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

def MSE_R2(poly_degrees, MSE, R2, N, title, fname):
    fig, axes = plt.subplots(1, 2)
    ax = axes[0]
    #fig.suptitle(title, fontsize=20)
    ax.set_title(f'Mean Square Error, {N}x{N}', fontsize=15)
    ax.plot(poly_degrees, MSE[0], label="Training data")
    ax.plot(poly_degrees, MSE[1], label="Test data")
    ax.set_xlabel('Polynomial Degree', fontsize=15)
    ax.set_ylabel('MSE', fontsize=15)
    ax.legend(fontsize=15)

    ax = axes[1]
    ax.set_title(f'R2 score, {N}x{N}', fontsize=15)
    ax.plot(poly_degrees, R2[0], label="Training data")
    ax.plot(poly_degrees, R2[1], label="Test data")
    ax.set_xlabel('Polynomial Degree', fontsize=15)
    ax.set_ylabel('R2', fontsize=15)
    ax.legend(fontsize=15)
    fig.savefig(fname)
    plt.close(fig)

def test_vs_train(poly_degrees, MSE, N, title, fname):
    poly_degrees_new = np.arange(1, len(poly_degrees), 2)
    fig, ax = plt.subplots()
    ax.set_title(title, fontsize=20)
    ax.set_xticks(poly_degrees_new)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_title(f'Training vs test data, MSE, {N}x{N} datapoints', fontsize=18)
    ax.plot(poly_degrees, MSE[0], label='MSE, Training')
    ax.plot(poly_degrees, MSE[1], label='MSE, Test')
    ax.set_xlabel('Polynomial Degree', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(fontsize=18)
    fig.savefig(fname)
    plt.close(fig)

def bias_variance(poly_degrees, MSE, bias, variance, N, title, fname):
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

def bias_variance_poly(poly_degrees, MSE, bias, variance, title, fname):
    fig, axes = plt.subplots(1, 2)
    fig.suptitle(title, fontsize=20)
    ax = axes[0]
    ax.plot(poly_degrees, variance[1], label='Variance')
    ax.set_xlabel('Polynomial Degree', fontsize=15)
    ax.legend(fontsize=15)

    ax = axes[1]
    ax.plot(poly_degrees, bias[1], label='Bias')
    ax.set_xlabel('Polynomial Degree', fontsize=15)
    ax.legend(fontsize=15)
    fig.savefig(fname)
    plt.close(fig)

def error_of_polydeg(poly_degrees, MSE, title, fname):
    fig, ax = plt.subplots()
    ax.set_title(title, fontsize=20)
    ax.plot(poly_degrees, MSE[0], label = 'MSE, train data')
    ax.plot(poly_degrees, MSE[1], label = 'MSE, test data')
    ax.set_xlabel('Polynomial Degree', fontsize=15)
    ax.set_ylabel('MSE', fontsize=15)
    ax.legend(fontsize=15)
    fig.savefig(fname)
    plt.close(fig)

def owncode_vs_sklearn(poly_degrees, MSE, MSE_sklearn, title, fname):
    fig, ax = plt.subplots()
    ax.set_title(title + ' with test data', fontsize=20)
    ax.plot(poly_degrees, MSE_sklearn, label = 'MSE, sklearn')
    ax.plot(poly_degrees, MSE[1], 'r--', label = 'MSE, own code')
    ax.set_xlabel('Polynomial Degree', fontsize=15)
    ax.set_ylabel('MSE', fontsize=15)
    ax.legend(fontsize=15)
    fig.savefig(fname)
    plt.close(fig)

def error_of_lambda(lambdas, error_lmbd, title, fname):
    fig, ax = plt.subplots()
    ax.set_title(title, fontsize=20)
    ax.plot(np.log10(lambdas), error_lmbd[0], label = 'MSE, train data')
    ax.plot(np.log10(lambdas), error_lmbd[1], label = 'MSE, test data')
    ax.set_xlabel(r'$\log_{10}(\lambda)$', fontsize=15)
    ax.set_ylabel('MSE', fontsize=15)
    ax.legend(fontsize=15)
    fig.savefig(fname)
    plt.close(fig)

def plot_terrain(terrain):
    fig, ax = plt.subplots()
    ax.set_title('Terrain over Norway 1', fontsize=20)
    plt.imshow(terrain, cmap='gray')
    ax.set_xlabel('X', fontsize=15)
    ax.set_ylabel('Y', fontsize=15)
    fig.savefig('plot/exercise6_terrain.pdf')
    plt.close(fig)

def variance_of_lambda(lambdas, var_lmbd, title, fname):
    fig, ax = plt.subplots()
    ax.set_title(title, fontsize=20)
    ax.plot(np.log10(lambdas), var_lmbd[0], label = 'variance, train data')
    ax.plot(np.log10(lambdas), var_lmbd[1], label = 'variance, test data')
    ax.set_xlabel(r'$\log_{10}(\lambda)$', fontsize=15)
    ax.set_ylabel('variance', fontsize=15)
    ax.legend(fontsize=15)
    fig.savefig(fname)
    plt.close(fig)

def bias_variance_of_lambda(lambdas, var, bias, title, fname):
    var_min = np.argmin(var[1])
    fig, ax = plt.subplots(1,2)
    fig.suptitle(title, fontsize=20)
    ax[0].set_title('variance')
    ax[0].plot(np.log10(lambdas), var[1], label = 'variance')
    ax[0].scatter(np.log10(lambdas[var_min]), var[1][var_min], label = r'$\lambda$ =' + f'{lambdas[var_min]:.2f}')
    ax[1].set_title('Bias')
    ax[1].plot(np.log10(lambdas), bias[1], label = 'Bias')
    ax[1].scatter(np.log10(lambdas[var_min]), bias[1][var_min], label = r'$\lambda$ =' + f'{lambdas[var_min]:.2f}')
    ax[0].set_xlabel(r'$\log_{10}(\lambda)$', fontsize=15)
    ax[1].set_xlabel(r'$\log_{10}(\lambda)$', fontsize=15)
    ax[0].legend(fontsize=15)
    ax[1].legend(fontsize=15)
    fig.savefig(fname)
    plt.close(fig)

def model_terrain(X, x, y, beta, degree, N):
    xx, yy = np.meshgrid(x,y)
    z_model = X@beta
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title(f'p={degree}')
    surf = ax.plot_surface(xx, yy, z_model, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(f'surf_plot_p{degree}_N{N}.png')

def CI(beta):
    diag_ele = np.zeros(len(beta))
    X_i = np.linalg.pinv(X.T@X)
    for i in range(len(beta)):
        diag_ele[i] = X_i[i,i]
    var_beta = var_error*diag_ele
    betas = np.arange(1, len(beta) + 1)
    CI_u = beta + 1.96*var_beta/np.sqrt(N)
    CI_l = beta - 1.96*var_beta/np.sqrt(N)
    yerr = 2*var_beta/np.sqrt(N)
    plt.fill_between(betas, CI_u, CI_l, alpha = 0.9)
    plt.scatter(betas, beta, marker = 'x', color = 'red', label = 'beta')
    plt.title(r'$\beta$ values with corresponding confidence interval of 95$\%$,' + f'p={degree}, n={N}')
    plt.xlabel(r'$\beta$')
    plt.xticks(betas)
    plt.ylabel(r'$\beta$ values')
    plt.legend()
    plt.savefig(f'conf_plot_p{degree}_N{N}.png')
