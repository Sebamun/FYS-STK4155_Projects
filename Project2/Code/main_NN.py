import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from sklearn.model_selection import train_test_split
import time

from methods_NN import Sigmoid, Tang_hyp, RELU, ELU, Leaky, Heaviside
from common import FrankeFunction, learning_schedule
from plots import plot_surface


f = open("../Textfiles/time_neural_network.txt", "w")
f.write('Model name| time | MSE  | lambda | learning rate | layers | number of neurons | batch-size | epochs   | \n')
f.write('------------------------------------------------------------------------------------------- \n')
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
n_layers_list = [1]#[1,2,3,4,5,6]
n_hidden_neurons_list = [5]#[5,10,20]
n_inputs = X.shape[0]
n_features = X.shape[1]
n_outputs = n_inputs

epochs = 3000
eta = 5e-4
lmbd_list = [5e-4] #[1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
gamma = 0.9
tol = 0.011
batch_size_list = [500]
t0, t1 = 0.1, 100 # Paramters used in learning rate.

start = time.time()
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=1)

sigmoid_model = Sigmoid(t0, t1, lmbd, gamma, n_layers, n_hidden_neurons, X_train, 'regression')
sigmoid_model.train(X_train, z_train, epochs, batch_size, learning_schedule=lambda t,t0,t1: t0)
time_sigmoid = time.time()
z_h, a_h, z_o_sigmoid, a_L = sigmoid_model.feed_forward(X_test)
MSE_sigmoid = np.mean((z_test - z_o_sigmoid)**2)
f.write(f'Sigmoid   |  {(time_sigmoid-start):.1f} | {MSE_sigmoid:.3f}|  {lmbd}  |     {eta}     |   {n_layers}    |        {n_hidden_neurons}         |   {epochs}   | \n')

plot_surface(X, sigmoid_model, 'Sigmoid', epochs, n_layers, xx, yy, N)
# plot_surface(X, TANH_model, 'tanh', epochs, n_layers, xx, yy, N)
f.close()
quit()

TANH_model = Tang_hyp(t0, t1, lmbd, gamma, n_layers, n_hidden_neurons, X_train, 'regression')
TANH_model.train(X_train, z_train, epochs, batch_size, learning_schedule=lambda t,t0,t1: t0)
z_h, a_h, z_o_tanh, a_L = TANH_model.feed_forward(X_test)
time_tanh = time.time()
MSE_tanh = np.mean((z_test - z_o_tanh)**2)
f.write(f'Tanh      | {(time_tanh-start):.1f} | {MSE_tanh:.3f}|  {lmbd}  |     {eta}     |   {n_layers}    |        {n_hidden_neurons}         |   {epochs}   | \n')



eta = 0.001 #Learning rate
RELU_model = RELU(t0, t1, lmbd, gamma, n_layers, n_hidden_neurons, X_train, 'regression')
RELU_model.train(X_train, z_train, epochs, batch_size, learning_schedule=lambda t,t0,t1: t0)
time_relu = time.time()
z_h, a_h, z_o_relu, a_L = RELU_model.feed_forward(X_test)
time_relu = time.time()
MSE_relu = np.mean((z_test - z_o_relu)**2)
f.write(f'Relu      | {(time_relu-start):.1f} | {MSE_relu:.3f}|  {lmbd}  |     {eta}     |   {n_layers}    |        {n_hidden_neurons}         |   {epochs}   | \n')

ELU_model = ELU(t0, t1, lmbd, gamma, n_layers, n_hidden_neurons, X_train, 'regression')
ELU_model.train(X_train, z_train, epochs, batch_size, learning_schedule=lambda t,t0,t1: t0)
z_h, a_h, z_o_elu, a_L = ELU_model.feed_forward(X_test)
time_elu = time.time()
MSE_elu = np.mean((z_test - z_o_elu)**2)
f.write(f'Elu       | {(time_elu-start):.1f} | {MSE_relu:.3f}|  {lmbd}  |     {eta}     |   {n_layers}    |        {n_hidden_neurons}         |   {epochs}   | \n')

HS_model = Heaviside(t0, t1, lmbd, gamma, n_layers, n_hidden_neurons, X_train, 'regression')
HS_model.train(X_train, z_train, epochs, batch_size, learning_schedule=lambda t,t0,t1: t0)
z_h, a_h, z_o_HS, a_L = HS_model.feed_forward(X_test)
time_HS = time.time()
MSE_HS = np.mean((z_test - z_o_HS)**2)
f.write(f'Heaviside | {(time_HS-start):.1f} | {MSE_relu:.3f}|  {lmbd}  |     {eta}     |   {n_layers}    |        {n_hidden_neurons}         |   {epochs}   | \n')

Leaky_model = Leaky(t0, t1, lmbd, gamma, n_layers, n_hidden_neurons, X_train, 'regression')
Leaky_model.train(X_train, z_train, epochs, batch_size, learning_schedule=lambda t,t0,t1: t0)
z_h, a_h, z_o_leaky, a_L = Leaky_model.feed_forward(X_test)
time_leaky = time.time()
MSE_leaky = np.mean((z_test - z_o_leaky)**2)
f.write(f'Leaky     | {(time_leaky-start):.1f} | {MSE_relu:.3f}|  {lmbd}  |     {eta}     |   {n_layers}    |        {n_hidden_neurons}         |   {epochs}   | \n')

f.close()

plot_surface(X, sigmoid_model, 'Sigmoid', epochs, n_layers, xx, yy, N)
plot_surface(X, TANH_model, 'tanh', epochs, n_layers, xx, yy, N)
plot_surface(X, RELU_model, 'RELU', epochs, n_layers, xx, yy, N)
plot_surface(X, ELU_model, 'ELU', epochs, n_layers, xx, yy, N)
plot_surface(X, Leaky_model, 'Leaky', epochs, n_layers, xx, yy, N)
plot_surface(X, HS_model, 'Heaviside', epochs, n_layers, xx, yy, N)
