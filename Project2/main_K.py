import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from sklearn.model_selection import train_test_split
import time
from methods_K import Sigmoid, Tang_hyp, RELU, ELU, Leaky, Heaviside
from common_K import FrankeFunction, initialize


start = time.time()
np.random.seed(1235)

#define the initial conditions for generating data
n = 20

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
n_layers = 1
n_hidden_neurons = 20
n_inputs = X.shape[0]
n_outputs = n_inputs



X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=1)


iterations = 80000
eta = 0.001

input_weights, hidden_weights, output_weights, hidden_bias, output_bias = initialize(n_layers, n_hidden_neurons, X, N)
sigmoid_model = Sigmoid(eta, n_layers, n_hidden_neurons, input_weights, hidden_weights,
                output_weights, hidden_bias, output_bias)
for i in range(iterations):
    sigmoid_model.back_propagation(X_train, z_train)
z_h, a_h, z_o_sigmoid = sigmoid_model.feed_forward(X_test)
time_sigmoid = time.time()
MSE_sigmoid = np.mean((z_test - z_o_sigmoid)**2)
print(f'MSE = {MSE_sigmoid}, Sigmoid')
print(f'It took {(time_sigmoid-start):.1f} seconds.')

z_h, a_h, z_o_sigmoid = sigmoid_model.feed_forward(X)
z_o_sigmoid = np.reshape(z_o_sigmoid, (n,n))
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'Surface plot of model using Sigmoid, {iterations:.1e} iterations', fontsize=25)
ax.set_zticklabels([])
ax.set_xticklabels([])
ax.set_yticklabels([])
surf = ax.plot_surface(xx, yy, z_o_sigmoid, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
ax.set_zlim(-0.10, 1.40)
bbox = fig.bbox_inches.from_bounds(1, 1, 8, 6)
bbox_inches=bbox
plt.savefig(f'Sigmoid_model_N{N}_it{iterations:.1e}_{n_layers}L.png', bbox_inches='tight')
plt.show()


eta = 0.0001 #Learning rate
iterations = 80000
input_weights, hidden_weights, output_weights, hidden_bias, output_bias = initialize(n_layers, n_hidden_neurons, X, N)
RELU_model = RELU(eta, n_layers, n_hidden_neurons, input_weights, hidden_weights,
                output_weights, hidden_bias, output_bias)
for i in range(iterations):
    RELU_model.back_propagation(X_train, z_train)
z_h, a_h, z_o_relu = RELU_model.feed_forward(X_test)
time_relu = time.time()
MSE_relu = np.mean((z_test - z_o_relu)**2)
print(f'MSE = {MSE_relu}, RELU')
print(f'It took {(time_relu - time_sigmoid):.1f} seconds.')

z_h, a_h, z_o_relu = RELU_model.feed_forward(X)
z_o_relu = np.reshape(z_o_relu, (n,n))
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'Surface plot of model using RELU, {iterations:.1e} iterations', fontsize=25)
ax.set_zticklabels([])
ax.set_xticklabels([])
ax.set_yticklabels([])
surf = ax.plot_surface(xx, yy, z_o_relu, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
ax.set_zlim(-0.10, 1.40)
bbox = fig.bbox_inches.from_bounds(1, 1, 8, 6)
bbox_inches=bbox
plt.savefig(f'RELU_model_N{N}_it{iterations:.1e}_{n_layers}L.png', bbox_inches='tight')
plt.show()


input_weights, hidden_weights, output_weights, hidden_bias, output_bias = initialize(n_layers, n_hidden_neurons, X, N)
TANH_model = Tang_hyp(eta, n_layers, n_hidden_neurons, input_weights, hidden_weights,
                output_weights, hidden_bias, output_bias)
for i in range(iterations):
    TANH_model.back_propagation(X_train, z_train)
z_h, a_h, z_o_tanh = TANH_model.feed_forward(X_test)
time_tanh = time.time()
MSE_tanh = np.mean((z_test - z_o_tanh)**2)
print(f'MSE = {MSE_tanh}, tanh')
print(f'It took {(time_tanh - time_relu):.1f} seconds.')

z_h, a_h, z_o_tanh = TANH_model.feed_forward(X)
z_o_tanh = np.reshape(z_o_tanh, (n,n))
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'Surface plot of model using tanh, {iterations:.1e} iterations', fontsize=25)
ax.set_zticklabels([])
ax.set_xticklabels([])
ax.set_yticklabels([])
surf = ax.plot_surface(xx, yy, z_o_tanh, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
ax.set_zlim(-0.10, 1.40)
bbox = fig.bbox_inches.from_bounds(1, 1, 8, 6)
bbox_inches=bbox
plt.savefig(f'tanh_model_N{N}_it{iterations:.1e}_{n_layers}L.png', bbox_inches='tight')
plt.show()


input_weights, hidden_weights, output_weights, hidden_bias, output_bias = initialize(n_layers, n_hidden_neurons, X, N)
ELU_model = ELU(eta, n_layers, n_hidden_neurons, input_weights, hidden_weights,
                output_weights, hidden_bias, output_bias)
for i in range(iterations):
    ELU_model.back_propagation(X_train, z_train)
z_h, a_h, z_o_elu = ELU_model.feed_forward(X_test)
time_elu = time.time()
MSE_elu = np.mean((z_test - z_o_elu)**2)
print(f'MSE = {MSE_elu}, ELU')
print(f'It took {(time_elu - time_tanh):.1f} seconds.')

z_h, a_h, z_o_elu = ELU_model.feed_forward(X)
z_o_elu = np.reshape(z_o_elu, (n,n))
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'Surface plot of model using elu, {iterations:.1e} iterations', fontsize=25)
ax.set_zticklabels([])
ax.set_xticklabels([])
ax.set_yticklabels([])
surf = ax.plot_surface(xx, yy, z_o_elu, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
ax.set_zlim(-0.10, 1.40)
bbox = fig.bbox_inches.from_bounds(1, 1, 8, 6)
bbox_inches=bbox
plt.savefig(f'ELU_model_N{N}_it{iterations:.1e}_{n_layers}L.png', bbox_inches='tight')
plt.show()


input_weights, hidden_weights, output_weights, hidden_bias, output_bias = initialize(n_layers, n_hidden_neurons, X, N)
HS_model = Heaviside(eta, n_layers, n_hidden_neurons, input_weights, hidden_weights,
                output_weights, hidden_bias, output_bias)
for i in range(iterations):
    HS_model.back_propagation(X_train, z_train)
z_h, a_h, z_o_HS = HS_model.feed_forward(X_test)
time_HS = time.time()
MSE_HS = np.mean((z_test - z_o_HS)**2)
print(f'MSE = {MSE_HS}, Heaviside')
print(f'It took {(time_HS - time_elu):.1f} seconds.')

z_h, a_h, z_o_HS = HS_model.feed_forward(X)
z_o_HS = np.reshape(z_o_HS, (n,n))
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'Surface plot of model using heaviside, {iterations:.1e} iterations', fontsize=25)
ax.set_zticklabels([])
ax.set_xticklabels([])
ax.set_yticklabels([])
surf = ax.plot_surface(xx, yy, z_o_HS, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
ax.set_zlim(-0.10, 1.40)
bbox = fig.bbox_inches.from_bounds(1, 1, 8, 6)
bbox_inches=bbox
plt.savefig(f'HS_model_N{N}_it{iterations:.1e}_{n_layers}L.png', bbox_inches='tight')
plt.show()

input_weights, hidden_weights, output_weights, hidden_bias, output_bias = initialize(n_layers, n_hidden_neurons, X, N)
Leaky_model = Leaky(eta, n_layers, n_hidden_neurons, input_weights, hidden_weights,
                output_weights, hidden_bias, output_bias)
for i in range(iterations):
    Leaky_model.back_propagation(X_train, z_train)
z_h, a_h, z_o_leaky = Leaky_model.feed_forward(X_test)
time_leaky = time.time()
MSE_leaky = np.mean((z_test - z_o_leaky)**2)
print(f'MSE = {MSE_leaky}, Leaky')
print(f'It took {(time_leaky - time_HS):.1f} seconds.')

z_h, a_h, z_o_leaky = Leaky_model.feed_forward(X)
z_o_leaky = np.reshape(z_o_leaky, (n,n))
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'Surface plot of model using Leaky RELU, {iterations:.1e} iterations', fontsize=25)
ax.set_zticklabels([])
ax.set_xticklabels([])
ax.set_yticklabels([])
surf = ax.plot_surface(xx, yy, z_o_leaky, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
ax.set_zlim(-0.10, 1.40)
bbox = fig.bbox_inches.from_bounds(1, 1, 8, 6)
bbox_inches=bbox
plt.savefig(f'Leaky_model_N{N}_it{iterations:.1e}_{n_layers}L.png', bbox_inches='tight')
plt.show()
