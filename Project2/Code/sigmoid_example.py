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

def tanh(x):
    return np.tanh(x)

def tanh_der(x):
    return -np.tanh(x)**2 + 1


def RELU(x):
    x[x>0] = x[x>0]
    x[x<0] = 0
    return x

def der_RELU(x):
    x[x>0] = 1
    x[x<0] = 0
    return x


x = np.linspace(-1, 1, 1000)
dx = x[1] - x[0]
def z(x,w):
    return x*w - w/2
w = 10
z0 = x
z1 = activation_func(z(z0, w))
z2 = activation_func(z(z1, w))
z3 = activation_func(z(z2, w))
dz1 = np.gradient(z1, dx)
dz2 = np.gradient(z2, dx)
dz3 = np.gradient(z3, dx)
fig, ax = plt.subplots()
ax.plot(x, dz1, label=r'$z = z_1 = w \cdot x - \frac{w}{2}$')
ax.plot(x, dz2, label=r'$z = w \cdot $ Sig($z_1$) $- \frac{w}{2}$')
# ax.plot(x, z1, label='Sigmoid1')
# ax.plot(x, z2, label='sigmoid2')
ax.tick_params(axis='both', which='major', labelsize=18)
ax.set_xlabel('x', fontsize=16)
ax.set_title(r'Derivative of Sig(z)', fontsize = 18)
plt.legend(prop={'size': 18})
plt.savefig('../Plots/sig_sig_der_example.png')

quit()


fig, ax = plt.subplots()
ax.plot(x, activation_func(x), label='Sigmoid1')
ax.plot(x, activation_func(activation_func(x)), label='Sigmoid2')
ax.plot(x, activation_func(activation_func(activation_func(x))), label='Sigmoid3')
ax.plot(x, der_act_func(x), label='Derivative of Sigmoid')
ax.tick_params(axis='both', which='major', labelsize=18)
ax.set_xlabel('x', fontsize=16)
plt.legend(prop={'size': 18})
plt.savefig('../Plots/sig_der_example.png')

# print(np.mean(activation_func(x)))
# print(activation_func(activation_func(100)))
# print(activation_func(activation_func(-100)))
# print(activation_func(activation_func(activation_func(100))))
# print(activation_func(activation_func(activation_func(-100))))
# print(activation_func(activation_func(activation_func(activation_func(100)))))
# print(activation_func(activation_func(activation_func(activation_func(-100)))))
# quit()

x = np.linspace(-2, 2, 10000)
dx = x[1] - x[0]
y = 3
y2 = 1

def z(x,y):
    return x*y

z0 = x
z1 = tanh(z(z0, y))
z2 = tanh(z(z1, y))
z3 = z(z2, y)

fig, ax = plt.subplots(1,2)
# ax.plot(x, z1, label=r'tanh($wx$)')
ax[0].plot(x, np.gradient(z1, dx), label=r'$w = 3$')
z0 = x
z1 = tanh(z(z0, y2))
ax[0].plot(x, np.gradient(z1, dx), label=r'$w = 1$')
ax[0].tick_params(axis='both', which='major', labelsize=18)
ax[0].set_xlabel('x', fontsize=16)
ax[0].set_title(r'Derivative of tanh($w \cdot x$)', fontsize=18)
ax[0].set_ylim(-0.1, 4)
ax[0].legend(loc='upper left', prop={'size': 16})
# plt.savefig('../Plots/tanh_der_example.png')


z0 = x
z1 = RELU(z(z0, y))
#ax.plot(x, z1, label=r'RELU($5 \cdot x$)')
ax[1].plot(x, np.gradient(z1, dx), label=r'$w = 3$')
z0 = x
z1 = RELU(z(z0, y2))
#ax.plot(x, z1, label=r'RELU($1 \cdot x$)')
ax[1].plot(x, np.gradient(z1, dx), label=r'$w = 1$')
ax[1].tick_params(axis='both', which='major', labelsize=18)
ax[1].set_xlabel('x', fontsize=16)
ax[1].set_title(r'Derivative of RELU($w \cdot x$)', fontsize=18)
ax[1].set_ylim(-0.1, 4)
ax[1].legend(loc='upper left', prop={'size': 16})
plt.savefig('../Plots/RELU_der_example.png')
