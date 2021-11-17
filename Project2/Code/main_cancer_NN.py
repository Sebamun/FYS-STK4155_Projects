import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.neural_network import MLPClassifier

from methods_NN import Sigmoid, Tang_hyp, RELU, ELU, Leaky, Heaviside
from plots import accuracy_epoch
from common import prepare_cancer_data

np.random.seed(1235)

X_train_scaled, X_test_scaled, y_train, y_test = prepare_cancer_data()

n_layers = 2
n_hidden_neurons = 20
n_features = X_train_scaled.shape[1]

n_epochs = [1, 10, 100, 1000, 10000, 100000, 1000000] # Number of epochs.
# n_epochs = [100] # Number of epochs.


M = 50
t0, t1 = 0.5, 100 # Paramters used in learning rate. # 50

lmbd = 0.12
gamma = 0.8

accuracy_test = np.zeros(len(n_epochs))
accuracy_train = np.zeros_like(accuracy_test)

model = Sigmoid(t0, t1, lmbd, gamma, n_layers,
                n_hidden_neurons, n_features, 'classification')
for i, epoch in enumerate(n_epochs):
    print(f'Iteration {i+1}/{len(n_epochs)}, with {n_layers} hidden layers.')
    print('---------------------------------------')

    model.train(X_train_scaled, y_train, epoch, M)

    *_, a_L_test = model.feed_forward(X_test_scaled, y_test)
    *_, a_L_train = model.feed_forward(X_train_scaled, y_train)

    accuracy_test[i] = np.mean(abs(a_L_test - y_test) < 0.5)
    print(f"Test set accuracy NN with own code is {accuracy_test[i]:.5f} for {epoch} epochs.")
    accuracy_train[i] = np.mean(abs(a_L_train - y_train) < 0.5)
    print(f"Train set accuracy NN with own code: {accuracy_train[i]:.5f} for {epoch} epochs.")
    print(" ")

    clf = MLPClassifier(hidden_layer_sizes=n_layers, activation='logistic',
                        batch_size=M, solver='sgd', alpha=lmbd, learning_rate='invscaling',
                        power_t=0.0005, max_iter=epoch, shuffle=True, random_state=1,
                        momentum=0.8).fit(X_train_scaled, y_train.ravel())

    accuracy_sklearn_test = clf.score(X_test_scaled, y_test)
    print(f"Test set accuracy NN with sklearn is {accuracy_sklearn_test:.5f} for {epoch} epochs.")
    accuracy_sklearn_train = clf.score(X_train_scaled, y_train)
    print(f"Train set accuracy NN with sklearn is {accuracy_sklearn_train:.5f} for {epoch} epochs.")

    cost_test = np.mean(-y_test*np.log(a_L_test)-(1-y_test)*np.log(1-a_L_test))
    cost_train = np.mean(-y_train*np.log(a_L_train)-(1-y_train)*np.log(1-a_L_train))
    print(" ")

    print(f"Cost for test set is {cost_test:.5f} for lambda = {lmbd}.")
    print(f"Cost for train set is {cost_train:.5f} for lambda = {lmbd}.")
    print(" ")

if len(n_epochs) > 2:
    print('Plotting')
    accuracy_epoch(n_epochs, accuracy_test, accuracy_train,
        r"Fit to cancer data using Neural Network, with $\lambda$ = " + f"{lmbd}",
        f"../Plots/NN_cancer_{lmbd}.pdf")


