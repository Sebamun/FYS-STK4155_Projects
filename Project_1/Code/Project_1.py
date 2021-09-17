from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from random import random, seed
import pandas as pd

def design_matrix(x,y,n): # Returns a design matrix given dimension of fit.
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2) / 2) # Number of elements in beta.
    X = np.ones((N,l)) # Creates array filled with ones.

    for i in range(n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = x**(i-k) * y**k
    return X

def FrankeFunction(x,y): # Returns value from frenke function + noise.
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 #+ np.random.normal(0,1,N)

#dx = 0.05 # Steplength.
#x = np.arange(0, 1, dx)
#y = np.arange(0, 1, dx)
#N = len(x)
np.random.seed(1234)
N = 1000
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
z = FrankeFunction(x,y)

poly_degrees = np.arange(1, 6)
MSE = np.zeros((2, len(poly_degrees)))
R2 = np.zeros((2, len(poly_degrees)))

for idx, degree in enumerate(poly_degrees): # (Test->20%, train->80%)
    X = design_matrix(x,y,n=degree) # Calls on function.
    a = np.linalg.matrix_rank(X) # We check if the matrix is singular.
    print(f'The rank of the matrix is {a} for polynomial of degree n={degree}.')
    #X_ = pd.DataFrame(X) # Converts matrix to dataframe for easier reading.
    #print(X_)

    # Split in training and test data:
    split_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, z, test_size = 0.2)

    X_train_new = np.delete(X_train, 0, 1) # We remove the first column (consisting of ones).
    X_test_new = np.delete(X_test, 0, 1)

    scaler = StandardScaler() # Scaling to unit variance.
    scaler.fit(X_train_new)
    scaler.fit(X_test_new)
    #print(pd.DataFrame(X_train_new))
    X_train_new = scaler.transform(X_train_new) # Subtracts mean and divide over standard deviation
    X_test_new = scaler.transform(X_test_new) # and make our data unitless.
    #print(pd.DataFrame(X_train_new))
    array_ones_train = np.ones((int(N*(1-split_size)), 1)) # 80%
    array_ones_test = np.ones((int(N*split_size), 1)) # 20%

    X_train = np.append(array_ones_train, X_train_new, axis = 1) # Here we create our new matrcies,
    X_test = np.append(array_ones_test, X_test_new, axis = 1) # Where the ones are appended in the first column.

    beta = np.linalg.pinv(X_train.T@X_train)@X_train.T@y_train # (.T takes the transpose of the matrix)
    y_predict_train = X_train@beta # (@ is matrix multiplication)
    y_predict_test = X_test@beta

    MSE[0, idx] = mean_squared_error(y_train, y_predict_train) #mean_squared_error(y_true, y_predict)
    MSE[1, idx] = mean_squared_error(y_test, y_predict_test)
    R2[0, idx] = r2_score(y_train, y_predict_train)
    R2[1, idx] = r2_score(y_test, y_predict_test)
# Ploting:
plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
fig, axes = plt.subplots(1, 2)
# Mean squared error:
ax = axes[0]
ax.set_title('Mean Square Error')
ax.plot(poly_degrees, MSE[0], label="Training data")
ax.plot(poly_degrees, MSE[1], label="Test data")
ax.set_xlabel('polynomial degree')
ax.set_ylabel('MSE')
ax.legend()
# R2 score:
ax = axes[1]
ax.set_title('R2 score')
ax.plot(poly_degrees, R2[0], label="Training data")
ax.plot(poly_degrees, R2[1], label="Test data")
ax.set_xlabel('polynomial degree')
ax.set_ylabel('R2')
ax.legend()
fig.savefig('plots/MSE_R2score_project1.pdf')

X, Y = np.meshgrid(x, y)
Z = FrankeFunction(X, Y)
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False) # To make 3D plot.
fig.add_axes(ax)
# Plot the surface:
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
# Customize the z axis:
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors:
fig.colorbar(surf, shrink=0.5, aspect=5)
fig.savefig('plots/surface_project1.pdf')
