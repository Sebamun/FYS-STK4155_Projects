import numpy as np

def der_MSE(y, y_o, _):
    return (y_o - y)

def crossEntropy(y, a):
    return np.mean(-y*np.log(a)-(1-y)*np.log(1-a))

def der_crossEntropy(y, y_o, x):
    val = np.sum((y_o - y)*x, axis = 1)
    return val.reshape(-1,1)

class NeuralNetwork:
    def __init__(self, t0, t1, lmbd, gamma, tol, n_layers, n_hidden_neurons, X_train, mode):
        self.t0, self.t1 = t0, t1
        self.lmbd = lmbd
        self.n_layers = n_layers
        self.n_hidden_neurons = n_hidden_neurons
        n_inputs = X_train.shape[0]
        n_features = X_train.shape[1]
        self.initialize_weights_and_biases(n_layers, n_hidden_neurons, n_inputs, n_features)
        self.v = np.zeros(5, dtype=object)
        self.gamma = gamma
        self.mode = mode
        self.tol = tol
        self.MSE = 2*tol
        if mode == 'regression':
            self.der_cost_func = der_MSE
        elif mode == 'classification':
            self.der_cost_func = der_crossEntropy

    def initialize_weights_and_biases(self, n_layers, n_hidden_neurons, n_inputs, n_features):
        if self.init_method() == "Random":
            self.input_weights = np.random.randn(n_features, n_hidden_neurons)
            self.hidden_weights = np.random.randn(n_layers - 1, n_hidden_neurons, n_hidden_neurons)
            self.hidden_bias = np.zeros((n_layers, n_hidden_neurons)) + 0.01
            self.output_weights = np.random.randn(n_hidden_neurons, 1)
            self.output_bias = np.zeros((1 , 1)) + 0.01

        elif self.init_method() == "Xavier":
            lim1 = np.sqrt(1/(n_inputs))
            self.input_weights = np.random.uniform(low = -lim1, high = lim1, size = (n_features, n_hidden_neurons))
            lim2 = np.sqrt(1/(n_hidden_neurons))
            self.hidden_weights = np.random.uniform(low = -lim2, high = lim2, size = (n_layers - 1, n_hidden_neurons, n_hidden_neurons))
            self.hidden_bias = np.zeros((n_layers, n_hidden_neurons)) #+ 0.01
            self.output_weights = np.random.uniform(low = -lim1, high = lim1, size = (n_hidden_neurons, 1))
            self.output_bias = np.zeros((1 , 1)) #+ 0.01

        elif self.init_method() == "Xavier_norm":
            lim1 = np.sqrt(1/(n_inputs+n_hidden_neurons))
            self.input_weights = np.random.uniform(low = -lim1, high = lim1, size = (n_features, n_hidden_neurons))
            lim2 = np.sqrt(6/(2*n_hidden_neurons))
            self.hidden_weights = np.random.uniform(low = -lim2, high = lim2, size = (n_layers - 1, n_hidden_neurons, n_hidden_neurons))
            self.hidden_bias = np.zeros((n_layers, n_hidden_neurons)) #+ 0.01
            self.output_weights = np.random.uniform(low = -lim1, high = lim1, size = (n_hidden_neurons, 1))
            self.output_bias = np.zeros((1 , 1)) #+ 0.01

        elif self.init_method() == "He":
            std1 = np.sqrt(2/(n_inputs))
            self.input_weights = np.random.normal(loc = 0, scale = std1, size = (n_features, n_hidden_neurons))
            std2 = np.sqrt(2/(n_hidden_neurons))
            self.hidden_weights = np.random.normal(loc = 0, scale = std2, size = (n_layers - 1, n_hidden_neurons, n_hidden_neurons))
            self.hidden_bias = np.zeros((n_layers, n_hidden_neurons)) #+ 0.01
            self.output_weights = np.random.normal(loc = 0, scale = std2, size = (n_hidden_neurons, 1))
            self.output_bias = np.zeros((1 , 1)) #+ 0.01



    def feed_forward(self, x):
        z_h = np.empty((self.n_layers, x.shape[0], self.n_hidden_neurons))
        a_h = np.empty((self.n_layers, x.shape[0], self.n_hidden_neurons))

        z_h[0] = x @ self.input_weights + self.hidden_bias[0]
        a_h[0] = self.activation_func(z_h[0])

        for i in range(1, self.n_layers):
            z_h[i] = a_h[i-1] @ self.hidden_weights[i-1] + self.hidden_bias[i]
            a_h[i] = self.activation_func(z_h[i])
            #NB! z_h[i-1] is one layer lower than w[i-1] and b[i-1] (see definition of z_h vs. b and w)
        z_o = a_h[-1] @ self.output_weights + self.output_bias
        if self.mode == "regression":
            a_L = z_o
        elif self.mode == "classification":
            a_L = Sigmoid.activation_func(self, z_o)

        return z_h, a_h, z_o, a_L

    def back_propagation(self, X, z, eta):
        z_h, a_h, z_o, a_L = self.feed_forward(X)
        self.MSE = np.mean((z - a_L)**2)
        #Calculate weight and bias gradients for output layer
        output_error = self.der_cost_func(z, a_L, a_h[-1])
        w_o_gradient = a_h[-1].T @ output_error + self.lmbd*self.output_weights
        b_o_gradient = np.sum(output_error, axis=0)

        w_h_gradient = np.zeros((self.n_layers - 1, self.n_hidden_neurons, self.n_hidden_neurons))\
         + self.lmbd*self.hidden_weights
        b_h_gradient = np.zeros((self.n_layers, self.n_hidden_neurons))

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
        self.update_weight_bias(w_o_gradient, b_o_gradient, w_h_gradient, b_h_gradient, w_i_gradient, eta)


    def update_weight_bias(self, w_o_gradient, b_o_gradient, w_h_gradient, b_h_gradient, w_i_gradient, eta):
        self.v[0] = self.gamma*self.v[0] + eta * w_o_gradient
        self.output_weights -= self.v[0]
        self.v[1] = self.gamma*self.v[1] + eta * b_o_gradient
        self.output_bias -= self.v[1]
        self.v[2] = self.gamma*self.v[2] + eta * w_h_gradient
        self.hidden_weights -= self.v[2]
        self.v[3] = self.gamma*self.v[3] + eta * b_h_gradient
        self.hidden_bias -= self.v[3]
        self.v[4] = self.gamma*self.v[4] + eta * w_i_gradient
        self.input_weights -= self.v[4]

    def train(self, X, z, epochs, batch_size, learning_schedule):
        """
        Train the neural network using SGD with momentum.
        input:
            X (array, shape=(N, number of features)): The input training data
            z (array, shape=(N, 1)): The target values for the training data
            epochs (int): The number of iterations
            batch_size (int): The number of inputs to use in each iteration
            learning_schedule (function(t, t0, t1)): How to calculate the learning rate during the SGD
        returns:
            None
        """
        N = int(X.shape[0])
        rng = np.random.default_rng(1234)
        indices = np.arange(N)
        for epoch in range(epochs+1):
            rng.shuffle(indices)
            X_s = X[indices]
            z_s = z[indices]
            if self.MSE < self.tol:
                print(f'Tolerance of {self.tol} reached at epoch={epoch}   | {self} | lmbd= {self.lmbd} | eta= {eta} | layers = {self.n_layers} | neurons = {self.n_hidden_neurons} |')
                break
            elif epoch == epochs:
                print(f'Tolerance not reached, stopped training at epoch={epoch}')
            for i in range(0, N, batch_size):
                eta = learning_schedule(epoch*(N/batch_size)+i, self.t0, self.t1)
                self.back_propagation(X_s[i:i+batch_size], z_s[i:i+batch_size], eta)

class Sigmoid(NeuralNetwork):
    def init_method(self):
        return "He"

    def __str__(self):
        return 'Sigmoid'

    def activation_func(self, z):
        return 1/(1 + np.exp(-z))

    def der_act_func(self, z):
        a = self.activation_func(z)
        return a*(1-a)

class Tang_hyp(NeuralNetwork):
    def init_method(self):
        return "Random"

    def __str__(self):
        return 'tanh'

    def activation_func(self, z):
        return np.tanh(z)

    def der_act_func(self, z):
        return -np.tanh(z)**2 + 1

class RELU(NeuralNetwork):
    def init_method(self):
        return "He"

    def __str__(self):
        return 'ReLU'

    def activation_func(self, z):
        a = z.copy()
        a[a<0] = 0
        return a

    def der_act_func(self, z):
        a = z.copy()
        a[a>0] = 1
        a[a<0] = 0
        return a

class ELU(NeuralNetwork):
    def init_method(self):
        return "He"

    def activation_func(self, z):
        a = z.copy()
        a[a<0] = np.exp(a[a<0])-1
        return a

    def der_act_func(self, z):
        a = z.copy()
        a[a>0] = 1
        a[a<0] = np.exp(a[a<0])
        return a

class Leaky(NeuralNetwork):
    def init_method(self):
        return "He"

    def activation_func(self, z):
        a = z.copy()
        a[a<0] = 0.1*a[a<0]
        return a

    def der_act_func(self, z):
        a = z.copy()
        a[a>0] = 1
        a[a<0] = 0.1
        return a

class Heaviside(NeuralNetwork):
    def init_method(self):
        return "Random"

    def activation_func(self, z):
        a = z.copy()
        a[a>0] = 1
        a[a<0] = 0
        return a

    def der_act_func(self, z):
        a = z.copy()
        a[:] = 0
        return a
