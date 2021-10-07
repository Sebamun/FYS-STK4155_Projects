import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

def MSE_R2(poly_degrees, MSE, R2, title, fname):
    fig, axes = plt.subplots(1, 2)
    ax = axes[0]
    fig.suptitle(title)
    ax.set_title('Mean Square Error')
    ax.plot(poly_degrees, MSE[0], label="Training data")
    ax.plot(poly_degrees, MSE[1], label="Test data")
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('MSE')
    ax.legend()

    ax = axes[1]
    ax.set_title('R2 score')
    ax.plot(poly_degrees, R2[0], label="Training data")
    ax.plot(poly_degrees, R2[1], label="Test data")
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('R2')
    ax.legend()
    fig.savefig(fname)
    plt.close(fig)

def bias_variance(poly_degrees, MSE, bias, variance, title, fname):
    fig, axes = plt.subplots(1, 2)
    fig.suptitle(title)
    ax = axes[0]
    ax.set_title('Bias variance tradeoff for training data')
    ax.plot(poly_degrees, MSE[0], label='MSE')
    ax.plot(poly_degrees, bias[0], label='Bias')
    ax.plot(poly_degrees, variance[0], label='Variance')
    ax.set_xlabel('Polynomial Degree')
    ax.legend()

    ax = axes[1]
    ax.set_title('Bias variance tradeoff for test data')
    ax.plot(poly_degrees, MSE[1], label='MSE')
    ax.plot(poly_degrees, bias[1], label='Bias')
    ax.plot(poly_degrees, variance[1], label='Variance')
    ax.set_xlabel('Polynomial Degree')
    ax.legend()
    fig.savefig(fname)
    plt.close(fig)

def error_of_polydeg(poly_degrees, MSE, title, fname):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.plot(poly_degrees, MSE[0], label = 'MSE, train data')
    ax.plot(poly_degrees, MSE[1], label = 'MSE, test data')
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('MSE')
    ax.legend()
    fig.savefig(fname)
    plt.close(fig)

def owncode_vs_sklearn(poly_degrees, MSE, MSE_sklearn, title, fname):
    fig, ax = plt.subplots()
    ax.set_title(title + ' with test data')
    ax.plot(
        poly_degrees, MSE_sklearn, label = 'MSE, sklearn'
    )
    ax.plot(
        poly_degrees, MSE[1], 'r--', label = 'MSE, own code'
    )
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('MSE')
    ax.legend()
    fig.savefig(fname)
    plt.close(fig)

def error_of_lambda(lambdas, error_lmbd, title, fname):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.plot(np.log10(lambdas), error_lmbd[0], label = 'MSE, train data')
    ax.plot(np.log10(lambdas), error_lmbd[1], label = 'MSE, test data')
    ax.set_xlabel(r'$\log_{10}(\lambda)$')
    ax.set_ylabel('MSE')
    ax.legend()
    fig.savefig(fname)
    plt.close(fig)

def plot_terrain(terrain):
    fig, ax = plt.subplots()
    ax.set_title('Terrain over Norway 1')
    plt.imshow(terrain, cmap='gray')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.savefig('plot/exercise6_terrain.pdf')
    plt.close(fig)
