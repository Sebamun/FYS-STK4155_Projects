import numpy as np
np.random.seed(1235)

def FrankeFunction(x,y):
    #Target function
    N = x.shape[0]
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + np.random.normal(0,0.1,(N,N))

def initialize(n_layers, n_hidden_neurons, n_features):
    #Define weight and-bias arrays for hidden and-output layers
    input_weights = np.random.randn(n_features, n_hidden_neurons)
    hidden_weights = np.empty((n_layers - 1, n_hidden_neurons, n_hidden_neurons))
    hidden_bias = np.empty((n_layers, n_hidden_neurons))

    for i in range(n_layers - 1):
        hidden_weights[i] = np.random.randn(n_hidden_neurons, n_hidden_neurons)
        hidden_bias[i] = np.zeros(n_hidden_neurons) + 0.1
    hidden_bias[-1] = np.zeros(n_hidden_neurons) + 0.1
    output_weights = np.random.randn(n_hidden_neurons, 1)
    output_bias = np.zeros((1 , 1)) + 0.1
    return input_weights, hidden_weights, output_weights, hidden_bias, output_bias



#
# def MSE(y_data, y_model):
#     """Calculate MSE"""
#     return np.mean((y_data - y_model)**2)
#
# def R2(y_data, y_model):
#     """Calculate R2 score"""
#     return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
