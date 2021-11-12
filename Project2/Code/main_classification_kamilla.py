import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from sklearn.model_selection import train_test_split
import time
from methods_kamilla import Sigmoid, Tang_hyp, RELU, ELU, Leaky, Heaviside
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

start = time.time()
np.random.seed(1235)

#define the initial conditions for generating data
cancer = load_breast_cancer()


inputs = cancer.data                     #Feature matrix of 569 rows (samples) and 30 columns (parameters)
outputs = cancer.target                  #Label array of 569 rows (0 for benign and 1 for malignant)
labels = cancer.feature_names[0:30]

X = inputs
Y = outputs

print('The content of the breast cancer dataset is:')      #Print information about the datasets
print(labels)
print('-------------------------')
print("inputs =  " + str(inputs.shape))
print("outputs =  " + str(outputs.shape))
print("labels =  "+ str(labels.shape))

plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


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

#Select features relevant to classification (texture,perimeter,compactness and symmetery)
#and add to input matrix
temp1 = np.reshape(X[:,1], (len(X[:,1]),1))
temp2 = np.reshape(X[:,2], (len(X[:,2]),1))
X = np.hstack((temp1,temp2))
print("Shape of design matrix = " + str(np.shape(X)))

Y = outputs.reshape(-1, 1)

#Generate training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# scaling the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

iterations = 100
batch_size = 10
eta = 0.001
n_layers = 1
n_hidden_neurons = 20
n_features = X.shape[1]

sigmoid_model = Sigmoid(eta, n_layers, n_hidden_neurons, n_features, 'classification')
sigmoid_model.train(X_train_scaled, y_train, iterations, batch_size)

y_h, a_h, y_o_sigmoid, a_L = sigmoid_model.feed_forward(X_test)
time_sigmoid = time.time()

# compare z_o_sigmoid and z_test
print(y_o_sigmoid.shape)
print(y_test.shape)
for i in range(len(y_o_sigmoid)):
    print(a_L[i], end='  ')
    print(y_test[i])

# True if z_test is 0 and z_o_sigmoid is less than 0.5,
# True if z_test is 1 and z_o_sigmoid is larger than 0.5
# else False
accuracy = np.mean(abs(a_L - y_test) < 0.5)
print(f'Accuracy = {accuracy}, Sigmoid, own code')

# Now with Sklearn
clf = MLPClassifier(random_state=1, max_iter = iterations).fit(X_train, y_train.ravel())
#clf = MLPClassifier(hidden_layer_sizes=(n_hidden_neurons), activation='logistic',alpha=lmbd, learning_rate_init=eta, max_iter=epochs)

#print(clf.predict_proba(X_test[:1]))
#print(clf.predict(X_test[:5, :]))

accuracy_sklearn = clf.score(X_test, y_test)
print(f'Accuracy = {accuracy_sklearn}, Sigmoid, sklearn')

end_time = time.time() - start
print(f'It took {(end_time):.1f} seconds.')

