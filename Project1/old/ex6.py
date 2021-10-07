# scipy.misc.imread
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy.random import normal, uniform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the terrain
terrain = imread('SRTM_data_Norway_1.tif')

N = 1000
m = 5 # polynomial order
terrain = terrain[:N,:N]

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

# Creates mesh of image pixels
x = np.linspace(0, 1, np.shape(terrain)[0])
y = np.linspace(0, 1, np.shape(terrain)[1])
x_mesh, y_mesh = np.meshgrid(x,y)

z = terrain

# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

poly_degrees = np.arange(1, 12)
z = np.ravel(z)
MSE = np.zeros((2, len(poly_degrees)))
R2 = np.zeros((2, len(poly_degrees)))
for idx, degree in enumerate(poly_degrees):
    X = create_X(x_mesh, y_mesh, n = degree)

    # split in training and test data
    split_size = 0.2
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.2)

    #Scaling:
    X_train_new = np.delete(X_train, 0, 1) # We remove the first column (consisting of ones).
    X_test_new = np.delete(X_test, 0, 1)

    scaler = StandardScaler() # Scaling to unit variance.
    scaler.fit(X_train_new)
    scaler.fit(X_test_new)

    X_train_new = scaler.transform(X_train_new) # Subtracts mean and divide over standard deviation
    X_test_new = scaler.transform(X_test_new) # and make our data unitless.

    array_ones_train = np.ones((int(N*(1-split_size)) * N, 1)) # 80%
    array_ones_test = np.ones((int(N*split_size) * N, 1)) # 20%
    X_train = np.append(array_ones_train, X_train_new, axis = 1) # Here we create our new matrcies,
    X_test = np.append(array_ones_test, X_test_new, axis = 1) # Where the ones are appended in the first column.

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
# fig.savefig('plots/exercise1_MSE_R2score.pdf')
plt.show()

"""
#Scaling:
X_train_new = np.delete(X_train, 0, 1) # We remove the first column (consisting of ones).
X_test_new = np.delete(X_test, 0, 1)

scaler = StandardScaler() # Scaling to unit variance.
scaler.fit(X_train_new)
scaler.fit(X_test_new)

X_train_new = scaler.transform(X_train_new) # Subtracts mean and divide over standard deviation
X_test_new = scaler.transform(X_test_new) # and make our data unitless.

array_ones_train = np.ones_like(X_train_new)#np.ones((int(N*(1-split_size)), 1)) # 80%
array_ones_test = np.ones_like(X_test_new)#np.ones((int(N*split_size), 1)) # 20%

size_train = len(array_ones_train)
size_test = len(array_ones_test)

X_train = np.append(np.ones((size_train, 1)), X_train_new, axis = 1) # Here we create our new matrcies,
X_test = np.append(np.ones((size_test, 1)), X_test_new, axis = 1) # Here we create our new matrcies,
"""