import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.neural_network import MLPClassifier

from methods_NN import Sigmoid, Tang_hyp, RELU, crossEntropy
from plots import accuracy_epoch
from common import prepare_cancer_data, learning_schedule, accuracy

np.random.seed(1235)

# setting the values of parameters that remain unchanged through the calculations
X_train, X_test, y_train, y_test = prepare_cancer_data()
n_layers = 2
n_hidden_neurons = 20
n_epochs = [1, 10, 100, 1000] #, 10000, 100000]
M = 50
t1 = 100
lmbd = 0.01
gamma = 0.8

def run_model_and_plot_results(model_class, t0, sklearn_activation,
        sklearn_learning_rate, plot_title, plot_fname):
    accuracy_test = np.zeros(len(n_epochs))
    accuracy_train = np.zeros_like(accuracy_test)

    model = model_class(t0, t1, lmbd, gamma, n_layers,
                    n_hidden_neurons, X_train, 'classification')

    for i, epoch in enumerate(n_epochs):
        print(f'Iteration {i+1}/{len(n_epochs)}, with {n_layers} hidden layers.')
        print('---------------------------------------')

        model.train(X_train, y_train, epoch, M, learning_schedule)

        *_, a_L_test = model.feed_forward(X_test, y_test)
        *_, a_L_train = model.feed_forward(X_train, y_train)

        accuracy_test[i] = accuracy(a_L_test, y_test)
        print(f"Test set accuracy NN with own code is {accuracy_test[i]:.5f} for {epoch} epochs.")
        accuracy_train[i] = accuracy(a_L_train, y_train)
        print(f"Train set accuracy NN with own code: {accuracy_train[i]:.5f} for {epoch} epochs.")
        print()

        clf = MLPClassifier(hidden_layer_sizes=n_layers, activation=sklearn_activation,
                            batch_size=M, solver='sgd', alpha=lmbd, learning_rate=sklearn_learning_rate,
                            learning_rate_init=t0, power_t=0.0005, max_iter=epoch, shuffle=True, random_state=1,
                            momentum=0.8).fit(X_train, y_train.ravel())

        print(f"Test set accuracy NN with sklearn is {clf.score(X_test, y_test):.5f} for {epoch} epochs.")
        print(f"Train set accuracy NN with sklearn is {clf.score(X_train, y_train):.5f} for {epoch} epochs.")
        print()

        print(f"Cost for test set is {crossEntropy(y_test, a_L_test):.5f} for lambda = {lmbd}.")
        print(f"Cost for train set is {crossEntropy(y_train, a_L_train):.5f} for lambda = {lmbd}.")
        print()

    if len(n_epochs) > 2:
        print('Plotting')
        accuracy_epoch(n_epochs, accuracy_test, accuracy_train, plot_title, plot_fname)

# # Sigmoid
# run_model_and_plot_results(
#     model_class=Sigmoid,
#     t0=0.5,
#     sklearn_activation='logistic',
#     sklearn_learning_rate='invscaling',
#     plot_title=r"Fit to cancer data using Neural Network, with $\lambda$ = " + f"{lmbd}",
#     plot_fname=f"../Plots/NN_cancer_Sigmoid_lmb_{lmbd}.pdf"
# )

# # RELU
# run_model_and_plot_results(
#     model_class=RELU,
#     t0 = 0.0001,
#     sklearn_activation='relu',
#     sklearn_learning_rate='constant',
#     plot_title=r"Fit to cancer data using Neural Network, with $\lambda$ = " + f"{lmbd}",
#     plot_fname=f"../Plots/NN_cancer_Sigmoid_lmb_{lmbd}.pdf"
# )

# Tanh
run_model_and_plot_results(
    model_class=Sigmoid,
    t0=0.5,
    sklearn_activation='tanh',
    sklearn_learning_rate='constant',
    plot_title=r"Fit to cancer data using Neural Network, with $\lambda$ = " + f"{lmbd}",
    plot_fname=f"../Plots/NN_cancer_Sigmoid_lmb_{lmbd}.pdf"
)