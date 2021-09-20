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
from random import random, seed

fig = plt.figure()
ax = fig.gca(projection='3d')
# Make data.
np.random.seed(1234)
N = 300
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
x, y = np.meshgrid(x,y)

def noise(N):
    rand_matrix = np.zeros((N,N))
    for i in range(N):
        rand_matrix[i] = np.random.normal(-0.03,0.03,N)
    return rand_matrix'

def create_X(x, y, n ):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)		# Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)

    return X

def FrankeFunction(x,y,noise):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    if(noise = True):
        return term1 + term2 + term3 + term4 + rand_matrix
    else:
        return term1 + term2 + term3 + term4 

z = FrankeFunction(x, y)
# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

poly_degrees = np.arange(1, 9)
MSE = np.zeros((2, len(poly_degrees)))
R2 = np.zeros((2, len(poly_degrees)))

for idx, degree in enumerate(poly_degrees):
    X = create_X(x, y, n = degree)
    z = np.ravel(z)
    # split in training and test data
    split_size = 0.19
    X_train, X_test, y_train, y_test = train_test_split(X, z, test_size = 0.19)

    # Ordinary least squares, NO TRAINING ON TEST DATA
    beta = np.linalg.pinv(X_train.T@X_train)@X_train.T@y_train  # train/fit the model

    y_predict_train = X_train@beta
    y_predict_test = X_test@beta

    #print(f'shape of y_train: {np.shape(y_train)}')
    #print(f'shape of y_predict_test: {np.shape(y_predict_test)}')

    MSE[0, idx] = mean_squared_error(y_train, y_predict_train)
    MSE[1, idx] = mean_squared_error(y_test, y_predict_test)
    R2[0, idx] = r2_score(y_train, y_predict_train)
    R2[1, idx] = r2_score(y_test, y_predict_test)


    #z_pred = np.reshape(y_predict_test, (15,15))
    z_pred = np.reshape(y_train, (270,270))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    xx = np.sort(np.random.uniform(0, 1, 270))
    yy = np.sort(np.random.uniform(0, 1, 270))
    xx, yy = np.meshgrid(xx,yy)

    surf = ax.plot_surface(xx, yy, z_pred, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


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
plt.show()
#fig.savefig('MSE_R2score_project1.pdf')
#print(y_predict_test)
