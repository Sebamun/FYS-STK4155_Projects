import numpy as np
from sklearn.linear_model import LogisticRegression

from methods_cancer_LR import LR
from plots import accuracy_epoch
from common import prepare_cancer_data

X_train_scaled, X_test_scaled, y_train, y_test = prepare_cancer_data()

n_epochs = [1, 10, 100, 1000, 10000, 100000, 1000000] # Number of epochs.
M = 50                  # Size of each minibatch (10 gave good results)
t0, t1 = 0.5, 100       # Paramters used in learning rate. # 50
m = int(len(X_train_scaled)/M) # Used when we split into minibatches.

lmbd = 0.15
gamma = 0.8 # Paramter used in momentum SGD.
v = 0 # Initial velocity for stochastic gradient descent with momentum.


model = LR(y_train, X_train_scaled, m, M) # Initialize our OLS model.

accuracy_test = np.zeros(len(n_epochs))
accuracy_train = np.zeros_like(accuracy_test)

for i, epoch in enumerate(n_epochs):
    print(f'Iteration {i+1}/{len(n_epochs)}')
    # Calculate methods for OLS:
    weights = model.SGD(epoch, lmbd, gamma, v, t0, t1)

    prediction_test = model.activation_func(X_test_scaled@weights)
    prediction_train = model.activation_func(X_train_scaled@weights)

    accuracy_test[i] = np.mean(abs(prediction_test - y_test) < 0.5)
    print(f"Test set accuracy LR with own code is {accuracy_test[i]:.5f} for {epoch} epochs.")
    accuracy_train[i] = np.mean(abs(prediction_train - y_train) < 0.5)
    print(f"Train set accuracy LR with own code: {accuracy_train[i]:.5f} for {epoch} epochs.")

    # Logistic Regression with sklearn
    logreg = LogisticRegression(solver='lbfgs', max_iter=epoch)
    logreg.fit(X_train_scaled, y_train.ravel())
    print(f"Test set accuracy LR with sklearn is {logreg.score(X_test_scaled,y_test):.5f} for {epoch} epochs.")
    print(f"Train set accuracy LR with sklearn is {logreg.score(X_train_scaled,y_train):.5f} for {epoch} epochs.")

print('Plotting')
accuracy_epoch(n_epochs, accuracy_test, accuracy_train,
                "Fit to cancer data with Logistic Regression",
                "../Plots/LG_cancer.pdf")
