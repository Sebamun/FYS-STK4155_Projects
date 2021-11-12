import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import numpy as np

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
