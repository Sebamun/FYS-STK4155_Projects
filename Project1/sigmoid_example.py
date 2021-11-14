import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

def activation_func(x):
    return 1/(1 + np.exp(-x))

def der_act_func(z):
    a = activation_func(z)
    return a*(1-a)


x = np.linspace(-20, 20, 1000)
plt.plot(x, activation_func(x), label='Sigmoid')
plt.plot(x, der_act_func(x), label='Derivative of Sigmoid')
plt.savefig('../Plots/sig_der_example.png')
