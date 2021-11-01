import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

def model_terrain(X, x, y, beta, N, title,z_data):
    z = X@beta
    z_model = np.reshape(z, (N, N))
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    bbox = fig.bbox_inches.from_bounds(1, 1, 8, 6)
    ax.set_zticklabels([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    surf = ax.plot_surface(x, y, z_model, cmap='binary',
    linewidth=0, antialiased=False)
    surf2 = ax.plot_surface(x, y, z_data, cmap=cm.coolwarm,
    linewidth=0, antialiased=False, alpha = 0.5)
    ax.set_title(title, fontsize=25)
    ax.zaxis.set_major_locator(LinearLocator(10))
    #plt.show()

def MSE_lamb(MSE1, MSE2, lamb):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot()
    ax.set_title('MSE as a function of lambda', fontsize=25)
    ax.plot(np.log(lamb), MSE1, label='SGD')
    ax.plot(np.log(lamb), MSE2, label='SGD with momentum')
    ax.legend()
