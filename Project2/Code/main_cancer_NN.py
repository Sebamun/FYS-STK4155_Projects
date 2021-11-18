import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.neural_network import MLPClassifier

from methods_NN import Sigmoid, Tang_hyp, RELU, ELU, Leaky, Heaviside
from plots import accuracy_epoch
from common import prepare_cancer_data

np.random.seed(1235)

X_train_scaled, X_test_scaled, y_train, y_test = prepare_cancer_data()

n_layers = 1
n_hidden_neurons = 10
n_features = X_train_scaled.shape[1]

n_epochs = [1, 10, 100, 1000, 10000, 100000, 1000000] # Number of epochs.
# n_epochs = [1000] # Number of epochs.
eta = 0.00001

M = 20
t0, t1 = 0.5, 100 # Paramters used in learning rate. # 50

lmbd = 0.0001
gamma = 0.9
tol = 0.0001
accuracy_test = np.zeros(len(n_epochs))
accuracy_train = np.zeros_like(accuracy_test)

model = Tang_hyp(eta, t1, lmbd, gamma, tol, n_layers,
                n_hidden_neurons, X_train_scaled, 'classification')
for i, epoch in enumerate(n_epochs):
    print(f'Iteration {i+1}/{len(n_epochs)}, with {n_layers} hidden layers.')

    model.train(X_train_scaled, y_train, epoch, M)

    *_, a_L_test = model.feed_forward(X_test_scaled)
    *_, a_L_train = model.feed_forward(X_train_scaled)

    accuracy_test[i] = np.mean(abs(a_L_test - y_test) < 0.5)
    print(f"Test set accuracy NN with own code is {accuracy_test[i]:.5f} for {epoch} epochs.")
    accuracy_train[i] = np.mean(abs(a_L_train - y_train) < 0.5)
    print(f"Train set accuracy NN with own code: {accuracy_train[i]:.5f} for {epoch} epochs.")

if len(n_epochs) > 2:
    print('Plotting')
    accuracy_epoch(n_epochs, accuracy_test, accuracy_train,
        "Fit to cancer data using Neural Network ",
        "../Plots/NN_cancer.pdf")

# # Now with Sklearn
clf = MLPClassifier(hidden_layer_sizes = n_layers,random_state = 1,\
activation = 'logistic', batch_size = M, learning_rate = 'invscaling',\
learning_rate_init = 0.001, max_iter = epochs,shuffle = True).fit(X_train, y_train.ravel())
#
#  # clf = MLPClassifier(hidden_layer_sizes=n_layers,random_state=1,\
#  #  solver='sgd', alpha=lmbd, activation='logistic', batch_size=batch_size, learning_rate='constant',\
#  #  learning_rate_init=0.001, power_t=0.5, max_iter=epochs, shuffle = True).fit(X_train, y_train.ravel())
#
# accuracy_sklearn = clf.score(X_test, y_test)
# print(f'Accuracy = {accuracy_sklearn}, sigmoid, sklearn')


#Visualisation of dataset (for correlation analysis)
# plt.figure()
# plt.scatter(X[:,0], X[:,2], s=40, c=Y, cmap=plt.cm.Spectral)
# plt.xlabel('Mean radius', fontweight='bold')
# plt.ylabel('Mean perimeter', fontweight='bold')
# plt.show()
#
# plt.figure()
# plt.scatter(X[:,5], X[:,6], s=40, c=Y, cmap=plt.cm.Spectral)
# plt.xlabel('Mean compactness', fontweight='bold')
# plt.ylabel('Mean concavity', fontweight='bold')
# plt.show()
#
#
# plt.figure()
# plt.scatter(X[:,0], X[:,1], s=40, c=Y, cmap=plt.cm.Spectral)
# plt.xlabel('Mean radius', fontweight='bold')
# plt.ylabel('Mean texture', fontweight='bold')
# plt.show()
#
# plt.figure()
# plt.scatter(X[:,2], X[:,1], s=40, c=Y, cmap=plt.cm.Spectral)
# plt.xlabel('Mean perimeter', fontweight='bold')
# plt.ylabel('Mean compactness', fontweight='bold')
# plt.show()
