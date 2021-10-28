import autograd.numpy as np
from autograd import elementwise_grad
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from common_sebastian import (MSE,  FrankeFunction, create_X
,model_terrain, scale, learning_schedule)

def cost_func_OLS(beta):
    return 1/(len(X)) * np.sum( (z - (X @ beta))**2 )

def cost_func_Ridge(beta):
    return 1/ len(X) * np.sum( (z - (X @ beta))**2 ) + np.sum(beta.T@beta) * lamb
    #2.0/n*X.T @ (X @ (beta)-z)+2*lmbda*beta

def analytical_gradient_OLS(beta):
    return 2.0/len(X)*X.T @ ((X @ beta)-z)

def analytical_gradient_Ridge(beta):
    return (2.0/len(X)*X.T @ (X @ (beta)-z)+2*lamb*beta)


np.random.seed(2018)
N = 10 # Number of datapoints.
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y)
z_data = z
n=5
z = np.ravel(z)[:,np.newaxis]
X = create_X(x,y,n)
X = scale(X)
#print(np.shape(X))

#Stochastic Gradient descent
def SGD(cost_func, analytical_gradient):
    beta = np.random.randn(np.shape(X)[1],1)
    n_epochs = 10000 # 10000
    M = 50   #size of each minibatch (10 gave good results)
    m = int(len(X)/M) #number of minibatches
    t0, t1 = 5, 50
    tol = 0.0001 # Tolerance for sum between analytical and numerical gradient.

    mean_squared_error_1 = np.zeros(n_epochs)

    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index*M:(random_index+1)*M]
            zi = z[random_index*M:(random_index+1)*M]
            gradients = elementwise_grad(cost_func)
            eta = learning_schedule(epoch*m+i,t0,t1)
            beta = beta - eta*gradients(beta)
            z_pred = X@beta
            mean_squared_error_1[epoch] = MSE(z, z_pred)
            if np.linalg.norm(gradients(beta))<0.01:
                break
    # Check for autograd function for last gradient:
    assert np.all(np.squeeze(analytical_gradient(beta))[:] \
    - np.squeeze(gradients(beta))[:] < tol)
    # Plotting:
    model_terrain(X,x,y,beta,N,'Stochastic gradient descent',z_data)
    return mean_squared_error_1


#Stochastic Gradient descent with momentum
def SGD_m(cost_func, analytical_gradient):

    beta = np.random.randn(np.shape(X)[1],1)
    n_epochs = 10000 # 10000
    M = 50   #size of each minibatch (10 gave good results)
    m = int(len(X)/M) #number of minibatches
    t0, t1 = 5, 50
    tol = 0.0001 # Tolerance for sum between analytical and numerical gradient.
    gamma = 0.9
    v = 0
    mean_squared_error_2 = np.zeros(n_epochs)

    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index*M:(random_index+1)*M]
            zi = z[random_index*M:(random_index+1)*M]
            gradients = elementwise_grad(cost_func)
            eta = learning_schedule(epoch*m+i,t0,t1)
            v = gamma*v - eta*gradients(beta)
            beta = beta + v
            z_pred = X@beta
            mean_squared_error_2[epoch] = MSE(z, z_pred)
            if np.linalg.norm(gradients(beta))<0.01:
                break

    # Check for autograd function for last gradient:
    assert np.all(np.squeeze(analytical_gradient(beta))[:] \
    - np.squeeze(gradients(beta))[:] < tol)
    # Plotting:
    model_terrain(X,x,y,beta,N,'Stochastic gradient descent',z_data)
    return mean_squared_error_2

lamb = 4.28*10**(-2)

MSE_SGD_ridge = SGD(cost_func_Ridge, analytical_gradient_Ridge)
MSE_SGDm_ridge = SGD_m(cost_func_Ridge, analytical_gradient_Ridge)
lamb = 0
MSE_SGD_OLS= SGD(cost_func_OLS, analytical_gradient_OLS)
MSE_SGDm_OLS = SGD_m(cost_func_OLS, analytical_gradient_OLS)

n_epochs = 10000
N_array = np.arange(0,n_epochs,1)

plt.plot(N_array, MSE_SGD_ridge,'r', label = 'MSE SGD ridge')
plt.plot(N_array, MSE_SGDm_ridge,'b', label = 'MSE SGD momentum ridge')
plt.plot(N_array, MSE_SGD_OLS,'g', label = 'MSE SGD OLS')
plt.plot(N_array, MSE_SGDm_OLS,'c', label = 'MSE SGD momentum OLS')
plt.legend()
plt.show()
