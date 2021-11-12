import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from sklearn.model_selection import train_test_split
import time
from methods_kamilla import Sigmoid, Tang_hyp, RELU, ELU, Leaky, Heaviside
from common_K import FrankeFunction, initialize


start = time.time()
np.random.seed(1235)

#define the initial conditions for generating data
n = 100

#Create 1-d input array with all linear combinations of x's and y's
x = np.linspace(0,1,n)
y = np.linspace(0,1,n)
N = int(n*n)
xx, yy = np.meshgrid(x,y)
x = np.ravel(xx)
y = np.ravel(yy)
X = np.zeros((len(x), 2))
X[:,0] = x
X[:,1] = y

#Produce target data
z = np.ravel(FrankeFunction(xx, yy))
z = np.reshape(z, (len(z), 1))

X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=1)

iterations = 1000
batch_size = 10
eta = 0.001
n_layers = 1
n_hidden_neurons = 20
n_features = X.shape[1]

sigmoid_model = Sigmoid(eta, n_layers, n_hidden_neurons, n_features, 'regression')
sigmoid_model.train(X_train, z_train, iterations, batch_size)
z_h, a_h, z_o_sigmoid, a_L = sigmoid_model.feed_forward(X_test)
time_sigmoid = time.time()
MSE_sigmoid = np.mean((z_test - a_L)**2)
print(f'MSE = {MSE_sigmoid}, Sigmoid')
print(f'It took {(time_sigmoid-start):.1f} seconds.')
#
# for i in range(len(a_L)):
#     print(a_L[i], end='  ')
#     print(z_test[i])

z_h, a_h, z_o_sigmoid, a_L = sigmoid_model.feed_forward(X)
z_o_sigmoid = np.reshape(a_L, (n,n))
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'Surface plot of model using Sigmoid, {iterations:.1e} iterations', fontsize=25)
ax.set_zticklabels([])
ax.set_xticklabels([])
ax.set_yticklabels([])
surf = ax.plot_surface(xx[::5], yy[::5], z_o_sigmoid[::5], cmap=cm.coolwarm,
linewidth=0, antialiased=False)
ax.set_zlim(-0.10, 1.40)
bbox = fig.bbox_inches.from_bounds(1, 1, 8, 6)
bbox_inches=bbox
plt.savefig(f'Sigmoid_model_N{N}_it{iterations:.1e}_{n_layers}L.png', bbox_inches='tight')
plt.show()