import autograd.numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

def MSE(y_data, y_model):
    """Calculate MSE"""
    return np.mean((y_data - y_model)**2)

def FrankeFunction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4
    #return 0.2*x + 0.7*y - 0.3*x*y

def create_X(x, y, n):
    """Create design Matrix"""
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2) # Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)

    return X

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
    linewidth=0, antialiased=False)
    ax.set_title(title, fontsize=25)
    ax.zaxis.set_major_locator(LinearLocator(10))
    plt.show()

def scale(X):
    """Scale all columns except the first (which is all ones)"""
    scaler = StandardScaler()
    scaler.fit(X[:, 1:])
    X[:, 1:] = scaler.transform(X[:, 1:])
    return X

def learning_schedule(t,t0,t1):
    """Learning rate used in SGD"""
    return t0/(t+t1)
