import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from sklearn.model_selection import train_test_split
import time
from methods_K import Sigmoid, Tang_hyp, RELU, ELU, Leaky, Heaviside
from common_K import FrankeFunction, initialize
from plot_K import plot_surface

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
n_layers = 1
n_hidden_neurons = 20
n_inputs = X.shape[0]
n_outputs = n_inputs
n_features = X.shape[1]

X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=1)

epochs = 1000
eta = 0.001
gamma = 0.9

sigmoid_model = Sigmoid(eta, n_layers, n_hidden_neurons, n_features, gamma)
sigmoid_model.train(X_train, z_train, epochs, 100)
time_sigmoid = time.time()
print(f'{epochs} Epochs took {(time_sigmoid-start):.1f} seconds.')
z_h, a_h, z_o_sigmoid = sigmoid_model.feed_forward(X_test)
MSE_sigmoid = np.mean((z_test - z_o_sigmoid)**2)
print(f'MSE = {MSE_sigmoid}, Sigmoid')


TANH_model = Tang_hyp(eta, n_layers, n_hidden_neurons, n_features, gamma)
TANH_model.train(X_train, z_train, epochs, 100)
z_h, a_h, z_o_tanh = TANH_model.feed_forward(X_test)
time_tanh = time.time()
print(f'{epochs} Epochs took {(time_sigmoid - time_tanh):.1f} seconds.')
MSE_tanh = np.mean((z_test - z_o_tanh)**2)
print(f'MSE = {MSE_tanh}, tanh')

eta = 0.001 #Learning rate
RELU_model = RELU(eta, n_layers, n_hidden_neurons, n_features, gamma)
RELU_model.train(X_train, z_train, epochs, 100)
time_relu = time.time()
print(f'{epochs} Epochs took {(time_tanh - time_relu):.1f} seconds.')
z_h, a_h, z_o_relu = RELU_model.feed_forward(X_test)
time_relu = time.time()
MSE_relu = np.mean((z_test - z_o_relu)**2)
print(f'MSE = {MSE_relu}, RELU')

ELU_model = ELU(eta, n_layers, n_hidden_neurons, n_features, gamma)
ELU_model.train(X_train, z_train, epochs, 100)
z_h, a_h, z_o_elu = ELU_model.feed_forward(X_test)
time_elu = time.time()
print(f'{epochs} Epochs took {(time_elu - time_tanh):.1f} seconds.')
MSE_elu = np.mean((z_test - z_o_elu)**2)
print(f'MSE = {MSE_elu}, ELU')

HS_model = Heaviside(eta, n_layers, n_hidden_neurons, n_features, gamma)
HS_model.train(X_train, z_train, epochs, 100)
z_h, a_h, z_o_HS = HS_model.feed_forward(X_test)
time_HS = time.time()
print(f'{epochs} Epochs took {(time_HS - time_elu):.1f} seconds.')
MSE_HS = np.mean((z_test - z_o_HS)**2)
print(f'MSE = {MSE_HS}, Heaviside')

Leaky_model = Leaky(eta, n_layers, n_hidden_neurons, n_features, gamma)
Leaky_model.train(X_train, z_train, epochs, 100)
z_h, a_h, z_o_leaky = Leaky_model.feed_forward(X_test)
time_leaky = time.time()
print(f'{epochs} Epochs took {(time_leaky - time_HS):.1f} seconds.')
MSE_leaky = np.mean((z_test - z_o_leaky)**2)
print(f'MSE = {MSE_leaky}, Leaky')

plot_surface(X, sigmoid_model, 'Sigmoid', epochs, n_layers, xx, yy, N)
plot_surface(X, TANH_model, 'tanh', epochs, n_layers, xx, yy, N)
plot_surface(X, RELU_model, 'RELU', epochs, n_layers, xx, yy, N)
plot_surface(X, ELU_model, 'ELU', epochs, n_layers, xx, yy, N)
plot_surface(X, Leaky_model, 'Leaky', epochs, n_layers, xx, yy, N)
plot_surface(X, HS_model, 'Heaviside', epochs, n_layers, xx, yy, N)
