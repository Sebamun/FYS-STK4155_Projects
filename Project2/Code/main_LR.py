from methods_LR import OLS
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

cancer = load_breast_cancer()


X = cancer.data                     #Feature matrix of 569 rows (samples) and 30 columns (parameters)
Y = cancer.target.reshape(-1, 1)            #Label array of 569 rows (0 for benign and 1 for malignant)
labels = cancer.feature_names[0:30]

#Generate training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# scaling the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

n_epochs = 10000 # Number of epochs. 10000
M = 10   #size of each minibatch (10 gave good results)
t0, t1 = 0.5, 100 # Paramters used in learning rate. # 50
m = int(len(X_train)/M) # Used when we split into minibatches.
lmbd = 0.4


model = OLS(y_train, X_train_scaled, m, M, lmbd) # Initialize our OLS model.

# Calculate methods for OLS:
Loss_OLS, weights = model.SGD(n_epochs, t0, t1, True)

prediction_test = model.activation_func(X_test_scaled@weights)
prediction_train = model.activation_func(X_train_scaled@weights)

# compare y_data and y_test
# print(prediction_test.shape)
# print(y_test.shape)

# for i in range(len(prediction_test)):
#     print(prediction_test[i], end='  ')
#     print(y_test[i])

accuracy_test = np.mean(abs(prediction_test - y_test) < 0.5)
print(f'Accuracy for test data = {accuracy_test}, Sigmoid, own code')

accuracy_train = np.mean(abs(prediction_train - y_train) < 0.5)
print(f'Accuracy for train data = {accuracy_train}, Sigmoid, own code')

# # Logistic Regression with sklearn
# logreg = LogisticRegression(solver='lbfgs')
# logreg.fit(X_train_scaled, y_train.ravel())
# print("Test set accuracy Logistic Regression with scaled data: {:.2f}".format(logreg.score(X_test_scaled,y_test)))
