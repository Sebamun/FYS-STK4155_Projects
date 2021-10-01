from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import sklearn.linear_model as skl

from common import FrankeFunction, create_X, MSE, scale

# For nice plots
plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

np.random.seed(2018)

# Making meshgrid of datapoints and compute Franke's function
N = 50
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
x, y = np.meshgrid(x, y)


z = FrankeFunction(x, y)

# Plot the surface.
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title(r'Surface for the Franke function with $x, y \in [0, 1]$ with noise')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.savefig('plots/exercise1_surface.pdf')
# plt.show()

z = FrankeFunction(x, y)
poly_degrees = np.arange(1, 10)
MSE = np.zeros((2, len(poly_degrees)))
R2 = np.zeros((2, len(poly_degrees)))
z = np.ravel(z)

for idx, degree in enumerate(poly_degrees):
    X = create_X(x, y, n = degree)

    # split in training and test data
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.2)

    # scale all columns except the first (which is all ones)
    X_train, X_test = scale(X_train, X_test)

    # Ordinary least squares, NO TRAINING ON TEST DATA
    beta = np.linalg.inv(X_train.T@X_train)@X_train.T@z_train  # train/fit the model
    z_predict_train = X_train@beta
    z_predict_test = X_test@beta

    MSE[0, idx] = mean_squared_error(z_train, z_predict_train)
    MSE[1, idx] = mean_squared_error(z_test, z_predict_test)
    R2[0, idx] = r2_score(z_train, z_predict_train)
    R2[1, idx] = r2_score(z_test, z_predict_test)


fig, axes = plt.subplots(1, 2)
ax = axes[0]
ax.set_title('Mean Square Error')
ax.plot(poly_degrees, MSE[0], label="Training data")
ax.plot(poly_degrees, MSE[1], label="Test data")
ax.set_xlabel('polynomial degree')
ax.set_ylabel('MSE')
ax.legend()

ax = axes[1]
ax.set_title('R2 score')
ax.plot(poly_degrees, R2[0], label="Training data")
ax.plot(poly_degrees, R2[1], label="Test data")
ax.set_xlabel('polynomial degree')
ax.set_ylabel('R2')
ax.legend()
fig.savefig('plots/exercise1_MSE_R2score.pdf')
