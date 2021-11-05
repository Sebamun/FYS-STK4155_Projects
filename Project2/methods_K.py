import numpy as np

class NeuralNetwork:
    def __init__(self, eta, n_layers, n_hidden_neurons, input_weights, hidden_weights,
                    output_weights, hidden_bias, output_bias):

        self.eta = eta
        self.n_layers = n_layers
        self.n_hidden_neurons = n_hidden_neurons
        self.input_weights = input_weights
        self.hidden_weights = hidden_weights
        self.output_weights = output_weights
        self.hidden_bias = hidden_bias
        self.output_bias = output_bias

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
        return z_h, a_h, z_o

    def back_propagation(self, X, z):
        z_h, a_h, z_o = self.feed_forward(X)
        #Calculate weight and bias gradients for output layer
        output_error = z_o-z
        w_o_gradient = a_h[-1].T @ output_error
        b_o_gradient = np.sum(output_error, axis=0)

        w_h_gradient = np.empty((self.n_layers - 1, self.n_hidden_neurons, self.n_hidden_neurons))
        b_h_gradient = np.empty((self.n_layers, self.n_hidden_neurons))


        hidden_error = output_error @ self.output_weights.T * self.der_act_func(z_h[-1])#a_h[-1]*(1-a_h[-1])#self.activation_func(z_h[-1])*(1-self.activation_func(z_h[-1]))#a_h[-1]*(1-a_h[-1])

        for i in reversed(range(1, self.n_layers - 1)):
            hidden_error = hidden_error @ self.hidden_weights[i].T * self.der_act_func(z_h[i])#a_h[-1]*(1-a_h[-1]) #self.activation_func(z_h[-1])*(1-self.activation_func(z_h[-1]))#a_h[i]*(1-a_h[i])
            w_h_gradient[i] = a_h[i-1].T @ hidden_error
            b_h_gradient[i] = np.sum(hidden_error, axis = 0)
        if self.n_layers > 1:
            input_error = hidden_error @ self.hidden_weights[0].T * self.der_act_func(z_h[0])#a_h[0]*(1-a_h[0])
        else:
            input_error = hidden_error
        w_i_gradient = X.T @ input_error
        b_h_gradient[0] = np.sum(input_error, axis = 0)


        #Update weights and biases
        self.output_weights -= self.eta*w_o_gradient
        self.output_bias -= self.eta*b_o_gradient
        self.hidden_weights -= self.eta * w_h_gradient
        self.hidden_bias -= self.eta * b_h_gradient
        self.input_weights -= self.eta * w_i_gradient


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
