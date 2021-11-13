import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from mpl_toolkits import mplot3d

plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

def model_terrain(X, x, y, beta, N, title, z_data, a, b):
    #ticks = np.linspace(0,1.5,5)
    z = X@beta
    z_model = np.reshape(z, (N, N))
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    bbox = fig.bbox_inches.from_bounds(1, 1, 8, 6)
    ax.tick_params(axis='x', which='major', labelsize=12)
    ax.tick_params(axis='y', which='major', labelsize=12)
    ax.tick_params(axis='z', which='major', labelsize=12)
    ax.xaxis.set_major_locator(LinearLocator(10))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel('x', fontsize=18)
    ax.yaxis.set_major_locator(LinearLocator(10))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_ylabel('y', fontsize=18)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_zlabel('z', fontsize=18)
    ax.view_init(25, 30) #(30,50)
    surf = ax.plot_surface(x, y, z_model, cmap='binary',
    linewidth=0, antialiased=False, alpha=a)
    surf2 = ax.plot_surface(x, y, z_data, cmap=cm.coolwarm,
    linewidth=0, antialiased=False, alpha = b)
    ax.set_title(title, fontsize=25)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    plt.savefig(f'../Plots/{title}')

def MSE_lamb(MSE, lamb, ind, optimal_lambda):
    # Plots the MSE as a function of lambda.
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot()
    ax.tick_params(axis='x', which='major', labelsize=18)
    ax.tick_params(axis='y', which='major', labelsize=18)
    ax.set_title('MSE as a function of lambda', fontsize=25)
    ax.plot(np.log(lamb), MSE, label='SGD with momentum')
    ax.plot(np.log(lamb[ind]), MSE[ind], markersize=20, marker = 'o', label=f'$\lambda$={lamb[ind]:.4f}')
    ax.plot(np.log(lamb[5]), MSE[5], markersize=20, marker= 'o', label=f'$\lambda$={optimal_lambda:.4f}')
    #ax.plot(np.log(optimal_lambda), optimal_MSE, markersize=20, marker= 'o', label=f'$\lambda$={optimal_lambda:.4f}')
    ax.set_xlabel(r'$\log_{10}(\lambda)$', fontsize=18)
    ax.set_ylabel('MSE', fontsize=18)
    ax.legend(fontsize=18)
    plt.savefig('../Plots/MSE_function_of_lambda')


def plot_surface(X, model, model_name, epochs, n_layers, xx, yy, N):
    z_h, a_h, z_o, a_L = model.feed_forward(X)
    n = int(np.sqrt(X.shape[0]))
    z_o = np.reshape(z_o, (n,n))
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'Surface plot of model using {model_name}, {epochs:.1e} iterations', fontsize=25)
    ax.set_zticklabels([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    surf = ax.plot_surface(xx, yy, z_o, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40)
    bbox = fig.bbox_inches.from_bounds(1, 1, 8, 6)
    bbox_inches=bbox
    plt.savefig(f'../Plots/{model_name}_model_N{N}_it{epochs:.1e}_{n_layers}L.png', bbox_inches='tight')



def accuracy_epoch(n_epochs, accuracy, title, fname, labl):
    fig, ax = plt.subplots()
    ax.set_title(title, fontsize=20)
    ax.plot(np.log10(n_epochs), accuracy, label=labl)
    ax.set_ylim(0.86, 1.0)
    ax.set_xlabel(r'$\log_{10}(\text{Number of epochs})$', fontsize=18)
    ax.set_ylabel('Accuracy score', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(fontsize=18)
    fig.savefig(f'../Plots/{fname}')
    plt.close(fig)
