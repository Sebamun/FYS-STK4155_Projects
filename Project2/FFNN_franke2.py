import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def FrankeFunction(x,y):
    #Target function
    N = x.shape[0]
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + np.random.normal(0,0.1,(N,N))

def sigmoid(x):
    #Activation function
    return 1/(1 + np.exp(-x))

def feed_forward(input):
    #Calculate Hidden layer
    z_h = np.matmul(hidden_weights, xy[:,0]) + hidden_bias
    a_h = sigmoid(z_h)
    #Calculate Output data
    z_o = np.matmul(output_weights, a_h) + output_bias
    #necessary (?) reshaping
    a_h = a_h[..., None].T
    z_o = z_o[..., None]
    return a_h, z_o, z_h

def backpropagation(xy, z, w_o, b_o):
    a_h, z_o, z_h = feed_forward(xy)
    z = z[..., None]
    #Calculate weight and-bias gradients for output layer
    output_error = z_o-z
    w_o_gradient = np.matmul(output_error, a_h)
    b_o_gradient = output_error
    #Calculate weight and-bias gradients for hidden layer
    hidden_error = np.reshape((output_error.T @ w_o), (n_hidden_neurons,)) * sigmoid(z_h)*(1-sigmoid(z_h))
    hidden_error = np.reshape(hidden_error, (n_hidden_neurons,1))
    w_h_gradient = hidden_error @ xy.T
    b_h_gradient = hidden_error
    return w_o_gradient, b_o_gradient, w_h_gradient, b_h_gradient

#define the initial conditions for generating data
np.random.seed(1234)
N = 20
#Create 1-d input array with all linear combinations of x's and y's
x = np.linspace(0,1,N)
y = np.linspace(0,1,N)
x_new = x[..., None]
y_new = x[..., None].T
xy = np.reshape(np.ravel(np.matmul(x_new, y_new)), (N*N,1))
x, y = np.meshgrid(x, y)

#Produce target data
z = np.ravel(FrankeFunction(x, y))

#Define number of neurons in hidden and ouput layer
n_hidden_neurons = int(xy.shape[0]/2)
n_inputs = xy.shape[0]
n_outputs = n_inputs

#Define weight and-bias arrays for hidden and-output layers
hidden_weights = np.random.randn(n_hidden_neurons, n_inputs)
hidden_bias = np.zeros(n_hidden_neurons) + 0.01

output_weights = np.random.randn(n_outputs, n_hidden_neurons)
output_bias = np.zeros(n_outputs) + 0.01

eta = 0.001 #Learning rate

iterations = 1000
for i in range(iterations):
    #Fetch gradients:
    dWo, dBo, dWh, dBh = backpropagation(xy, z, output_weights, output_bias)
    #necessary (?) reshaping
    dBo = np.reshape(dBo, (n_inputs,))
    dBh = np.reshape(dBh, (n_hidden_neurons,))
    #Update weights and biases
    output_weights -= eta*dWo
    output_bias -= eta*dBo
    hidden_weights -= eta * dWh
    hidden_bias -= eta * dBh

#Produce model output with improved weights and biases
a_h, z_o, z_h = feed_forward(xy)

#Plot dat shit
z_o = np.reshape(z_o, (N,N))
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'Surface plot of model', fontsize=25)
ax.set_zticklabels([])
ax.set_xticklabels([])
ax.set_yticklabels([])
surf = ax.plot_surface(x, y, z_o, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
ax.set_zlim(-0.10, 1.40)
bbox = fig.bbox_inches.from_bounds(1, 1, 8, 6)
bbox_inches=bbox
plt.savefig(f'FFNN_terrain_N{N}_it{iterations}.png', bbox_inches='tight')
plt.show()
