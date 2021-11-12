import numpy as np

def der_MSE(y, y_o, _):
    return (y_o - y)

def der_crossEntropy(y, y_o, x):
    val = np.mean((y_o - y)*x, axis = 1)
    return(val.reshape(-1,1))


class NeuralNetwork:
    def __init__(self, eta, lmbd, gamma, n_layers, n_hidden_neurons, n_features, mode):

        self.eta = eta
        self.lmbd = lmbd
        self.n_layers = n_layers
        self.n_hidden_neurons = n_hidden_neurons
        self.input_weights, self.hidden_weights, self.output_weights,\
        self.hidden_bias, self.output_bias = self.initialize(n_layers, n_hidden_neurons, n_features)
        self.v = np.zeros(5, dtype=object)
        self.gamma = gamma
        self.mode = mode
        if mode == 'regression':
            self.der_cost_func = der_MSE
        elif mode == 'classification':
            self.der_cost_func = der_crossEntropy

    def initialize(self, n_layers, n_hidden_neurons, n_features):
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

    def feed_forward(self, input):
        z_h = np.empty((self.n_layers, input.shape[0], self.n_hidden_neurons))
        a_h = np.empty((self.n_layers, input.shape[0], self.n_hidden_neurons))

        z_h[0] = input @ self.input_weights + self.hidden_bias[0]
        a_h[0] = self.activation_func(z_h[0])

        for i in range(1, self.n_layers):
            z_h[i] = a_h[i-1] @ self.hidden_weights[i-1] + self.hidden_bias[i-1]
            a_h[i] = self.activation_func(z_h[i])
            #NB! z_h[i-1] is one layer lower than w[i-1] and b[i-1] (see definition of z_h vs. b and w)
        z_o = a_h[-1] @ self.output_weights + self.output_bias
        if self.mode == "regression":
            a_L = z_o
        elif self.mode == "classification":
            a_L = self.activation_func(z_o)
        return z_h, a_h, z_o, a_L

    def back_propagation(self, X, z):
        z_h, a_h, z_o, a_L = self.feed_forward(X)
        #Calculate weight and bias gradients for output layer
        output_error = self.der_cost_func(z, a_L, a_h[-1])
        w_o_gradient = a_h[-1].T @ output_error + self.lmbd*self.output_weights
        b_o_gradient = np.sum(output_error, axis=0)

        w_h_gradient = np.empty((self.n_layers - 1, self.n_hidden_neurons, self.n_hidden_neurons))\
         + self.lmbd*self.hidden_weights
        b_h_gradient = np.empty((self.n_layers, self.n_hidden_neurons))

        hidden_error = output_error @ self.output_weights.T * self.der_act_func(z_h[-1])#a_h[-1]*(1-a_h[-1])#self.activation_func(z_h[-1])*(1-self.activation_func(z_h[-1]))#a_h[-1]*(1-a_h[-1])

        for i in reversed(range(1, self.n_layers - 1)):
            hidden_error = hidden_error @ self.hidden_weights[i].T * self.der_act_func(z_h[i])#a_h[-1]*(1-a_h[-1]) #self.activation_func(z_h[-1])*(1-self.activation_func(z_h[-1]))#a_h[i]*(1-a_h[i])
            w_h_gradient[i] = a_h[i-1].T @ hidden_error + self.lmbd*self.hidden_weights[i]
            b_h_gradient[i] = np.sum(hidden_error, axis = 0)
        if self.n_layers > 1:
            input_error = hidden_error @ self.hidden_weights[0].T * self.der_act_func(z_h[0])#a_h[0]*(1-a_h[0])
        else:
            input_error = hidden_error
        w_i_gradient = X.T @ input_error + self.lmbd*self.input_weights
        b_h_gradient[0] = np.sum(input_error, axis = 0)

        #Update weights and biases
        self.update_weight_bias(w_o_gradient, b_o_gradient, w_h_gradient, b_h_gradient, w_i_gradient)


    def update_weight_bias(self, w_o_gradient, b_o_gradient, w_h_gradient, b_h_gradient, w_i_gradient):

        self.v[0] = self.gamma*self.v[0] + self.eta * w_o_gradient
        self.output_weights -= self.v[0]
        self.v[1] = self.gamma*self.v[1] + self.eta * b_o_gradient
        self.output_bias -= self.v[1]
        self.v[2] = self.gamma*self.v[2] + self.eta * w_h_gradient
        self.hidden_weights -= self.v[2]
        self.v[3] = self.gamma*self.v[3] + self.eta * b_h_gradient
        self.hidden_bias -= self.v[3]
        self.v[4] = self.gamma*self.v[4] + self.eta * w_i_gradient
        self.input_weights -= self.v[4]


    def train(self, X, z, epochs, M):
        N = int(X.shape[0])
        rng = np.random.default_rng(1234)
        indices = np.arange(N)
        batch_size = M
        for i in range(epochs):
            rng.shuffle(indices)
            X_s = X[indices]
            z_s = z[indices]
            for i in range(0, N, batch_size):
                self.back_propagation(X_s[i:i+batch_size], z_s[i:i+batch_size])


class Sigmoid(NeuralNetwork):
    def activation_func(self, x):
        return 1/(1 + np.exp(-x))

    def der_act_func(self, z):
        a = self.activation_func(z)
        return a*(1-a)

class Tang_hyp(NeuralNetwork):
    def activation_func(self, x):
        return np.tanh(x)

    def der_act_func(self, x):
        return -np.tanh(x)**2 + 1

class RELU(NeuralNetwork):
    def activation_func(self, x):
        x[x>0] = x[x>0]
        x[x<0] = 0
        return x

    def der_act_func(self, x):
        x[x>0] = 1
        x[x<0] = 0
        return x

class ELU(NeuralNetwork):
    def activation_func(self, x):
        x[x<0] = np.exp(x[x<0])-1
        return x

    def der_act_func(self, x):
        x[x>0] = 1
        x[x<0] = np.exp(x[x<0])
        return x

class Leaky(NeuralNetwork):
    def activation_func(self, x):
        x[x>0] = x[x>0]
        x[x<0] = 0.1*x[x<0]
        return x

    def der_act_func(self, x):
        x[x>0] = 1
        x[x<0] = 0.1
        return x

class Heaviside(NeuralNetwork):
    def activation_func(self, x):
        x[x>0] = 1
        x[x<0] = 0
        return x

    def der_act_func(self, x):
        x[:] = 0
        return x
