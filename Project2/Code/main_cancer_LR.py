import numpy as np
from sklearn.linear_model import LogisticRegression
from methods_cancer_LR import Sigmoid
from methods_NN import crossEntropy
from plots import accuracy_epoch
from common import prepare_cancer_data, learning_schedule, accuracy

np.random.seed(1235)

# Setting the values of parameters that remain unchanged through the calculations
X_train, X_test, y_train, y_test = prepare_cancer_data()
n_epochs = [1, 10, 100, 1000, 10000, 100000, 1000000] # Number of epochs.
M = 20                  # Size of each minibatch (10 gave good results)
t0, t1 = 0.5, 100       # Paramters used in learning rate.
m = int(len(X_train)/M) # Used when we split into minibatches.
lmbd = 0.05
gamma = 0.8             # Paramter used in momentum SGD.
v = 0                   # Initial velocity for SGD with momentum.


def run_model_and_plot_results(model_class, sklearn_activation,
        sklearn_learning_rate, plot_title, plot_fname):
    accuracy_test = np.zeros(len(n_epochs))
    accuracy_train = np.zeros_like(accuracy_test)

    model = model_class(y_train, X_train, m, M)

    for i, epoch in enumerate(n_epochs):
        print(f'Iteration {i+1}/{len(n_epochs)}')
        print('---------------------------------------')
        # Calculate methods for OLS:
        weights = model.SGD(epoch, lmbd, gamma, v, t0, t1)

        prediction_test = model.activation_func(X_test@weights)
        prediction_train = model.activation_func(X_train@weights)

        accuracy_test[i] = accuracy(prediction_test, y_test)
        print(f"Test set accuracy LR with own code is {accuracy_test[i]:.5f} for {epoch} epochs.")
        accuracy_train[i] = accuracy(prediction_train, y_train)
        print(f"Train set accuracy LR with own code: {accuracy_train[i]:.5f} for {epoch} epochs.")
        print()

        # Logistic Regression with sklearn
        logreg = LogisticRegression(solver='lbfgs', max_iter=epoch)
        logreg.fit(X_train, y_train.ravel())

        print(f"Test set accuracy LR with sklearn is {logreg.score(X_test,y_test):.5f} for {epoch} epochs.")
        print(f"Train set accuracy LR with sklearn is {logreg.score(X_train,y_train):.5f} for {epoch} epochs.")
        print()

        print(f"Cost for test set is {crossEntropy(y_test, prediction_test):.5f} for lambda = {lmbd}.")
        print(f"Cost for train set is {crossEntropy(y_train, prediction_train):.5f} for lambda = {lmbd}.")
        print()

    if len(n_epochs) > 5:
        print('Plotting')
        accuracy_epoch(n_epochs, accuracy_test, accuracy_train, plot_title, plot_fname)


# Sigmoid
run_model_and_plot_results(
    model_class=Sigmoid,
    sklearn_activation='logistic',
    sklearn_learning_rate='invscaling',
    plot_title=r"Fit to cancer data using Logistic Regression, with Sigmoid and $\lambda$ = " + f"{lmbd}",
    plot_fname=f"../Plots/LR_cancer_sigmoid_lmb_{lmbd}.pdf"
)
